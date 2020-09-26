import re
import io
import os
import sys
import json
import time
import discord
import threading
import logging
import functools
from gpt2_server_sessions import gpt2_server_sessions
from datetime import datetime, timedelta
from discord.ext import commands
from discord import utils
from discord.ext.commands import has_permissions, MissingPermissions
from src import model, sample, encoder
import numpy as np
import tensorflow as tf

class GPT2Bot(commands.Cog):

    def __init__(self, bot):
        logging.basicConfig(level=logging.INFO)

        self.bot = bot
        self.not_ready_s = "Bot has not been initialized. Please type !init to initialize the bot."
        self.is_interfering = True
        self.not_ready = True
        self.sizeLimit=1000 # NOTE: Set this according to your own machine.
        self.guildIdList = []
        self.serverSessions = {}
        self.is_interfering = False
        self.models = os.listdir(os.path.join('models'))

    @commands.command()
    async def init(self, ctx):
        guilds = await self.bot.fetch_guilds(limit=150).flatten()
        for guild in guilds:
            self.guildIdList.append(guild.id)
        self.serverSessions = {}
        for serverid in self.guildIdList:
            self.serverSessions[serverid] = gpt2_server_sessions(serverid)
        await ctx.send("GPT-2 AI initialized")
        self.not_ready = False

    @commands.command()
    @commands.guild_only()
    async def talk(self, ctx, *, message):
        logging.info('MSG: ' + message)
        if (self.is_interfering):
            await ctx.send('Currently talking to someone. Try again later.')
            return
        if (self.not_ready):
            await ctx.send(self.not_ready_s)
            return
        server_id = ctx.message.guild.id
        logging.info('Guild: ' + str(server_id))
        self.is_interfering = True
        if message:
            context_tokens = self.serverSessions[server_id].enc.encode(message)
        for _ in range(self.serverSessions[server_id].nsamples):
            async with ctx.typing():
                start = time.time()
                if message:
                    text_generator = functools.partial(self.generate_text, server_id, context_tokens)
                    out = await self.bot.loop.run_in_executor(None, text_generator)
                else:
                    text_generator = functools.partial(self.generate_uncon_text, server_id)
                    out = await self.bot.loop.run_in_executor(None, text_generator)
                response = message + self.serverSessions[server_id].enc.decode(out[0])
                logging.info('RESPONSE GENERATED IN :' + str(round(time.time() - start, 2)) + ' seconds.')
                logging.info('RESPONSE: ' + response)
                logging.info('RESPONSE LEN: ' + str(len(response)))

                response_chunk = 0
                chunk_size = 1990
                if (len(response) > 2000):
                    while (len(response) > response_chunk):
                        await ctx.send(response[response_chunk:response_chunk + chunk_size])
                        response_chunk += chunk_size
                else:
                    await ctx.send(response)

        self.is_interfering = False

    def generate_text(self, server_id, context_tokens):
        return self.serverSessions[server_id].generate_text(context_tokens)

    def generate_uncon_text(self, server_id):
        return self.serverSessions[server_id].generate_uncon_text()

    @commands.command()
    @commands.guild_only()
    @commands.has_permissions(manage_messages=True)
    async def debugtalk(self, ctx, *, message):
        logging.info('MSG: ' + message)
        if (self.is_interfering):
            await ctx.send('Currently talking to someone. Try again later.')
            return
        if (self.not_ready):
            await ctx.send(self.not_ready_s)
            await ctx.send("Bot isn't ready yet.")
            return
        server_id = ctx.message.guild.id
        await ctx.send('```Guild: ' + str(server_id) + '\n'
            'Message received, generating response...```')
        logging.info('Guild: ' + str(server_id))
        self.is_interfering = True
        if message:
            context_tokens = self.serverSessions[server_id].enc.encode(message)
        for _ in range(self.serverSessions[server_id].nsamples):
            async with ctx.typing():
                start = time.time()
                if message:
                    text_generator = functools.partial(self.generate_text, server_id, context_tokens)
                    out = await self.bot.loop.run_in_executor(None, text_generator)
                else:
                    text_generator = functools.partial(self.generate_uncon_text, server_id)
                    out = await self.bot.loop.run_in_executor(None, text_generator)
                response = message + self.serverSessions[server_id].enc.decode(out[0])
                logging.info('RESPONSE GENERATED IN:' + str(round(time.time() - start, 2)) + ' SECONDS')
                logging.info('RESPONSE: ' + response)
                logging.info('RESPONSE LEN: ' + str(len(response)))


                response_chunk = 0
                chunk_size = 1990
                if (len(response) > 2000):
                    while (len(response) > response_chunk):
                        await ctx.send(response[response_chunk:response_chunk + chunk_size])
                        response_chunk += chunk_size
                        await ctx.send('```Response generated in: ' + str(round(time.time() - start, 2)) + ' seconds.\n'
                            'Response length: ' + str(len(response)) + '```')
                else:
                    await ctx.send(response)
                    await ctx.send('```Response generated in: ' + str(round(time.time() - start, 2)) + ' seconds.\n'
                        'Response length: ' + str(len(response)) + '```')

        self.is_interfering = False

    def generate_text(self, server_id, context_tokens):
        return self.serverSessions[server_id].generate_text(context_tokens)

    def generate_uncon_text(self, server_id):
        return self.serverSessions[server_id].generate_uncon_text()

    @commands.command()
    @commands.guild_only()
    @commands.has_permissions(manage_messages=True)
    async def getconfig(self, ctx):
        if (self.not_ready):
            await ctx.send(self.not_ready_s)
            return
        logging.info('CURRENT STATE.')
        server_id = ctx.message.guild.id
        await ctx.send('**Current state:**\n```'
            'N Samples: ' + str(self.serverSessions[server_id].nsamples) + "\n"
            'Max Length: ' + str(self.serverSessions[server_id].length) + "\n"
            'Temperature: ' + str(self.serverSessions[server_id].temperature) + "\n"
            'Top K: ' + str(self.serverSessions[server_id].top_k) + "\n"
            'Model: ' + str(self.serverSessions[server_id].model_name) + "```")

    @commands.command()
    @commands.guild_only()
    @commands.has_permissions(manage_messages=True)
    async def helpconfig(self, ctx):
        if (self.not_ready):
            await ctx.send(self.not_ready_s)
            return
        logging.info('HELP INVOKED.')
        await ctx.send('Configure the bot session by typing: `!setconfig <nsamples> <length> <temperature> <topk> <model>`.\n'
            '`nsamples` = Number of samples to generate\n'
            '`length = Estimate on how much to generate after your prompt`\n'
            '`temperature` = Lower temperature results in less random completions. As the temperature approaches zero, '
            'the model will become deterministic and repetitive. Higher temperature results in more random completions.\n'
            '`topk` = Integer value controlling diversity. 1 means only 1 word is considered for each step (token), '
            'resulting in deterministic completions, while 40 means 40 words are considered at each step. '
            '0 is a special setting meaning no restrictions. 40 generally is a good value.\n'
            '`model` = Set which model is used for generating text. The larger the model, the longer it will take to generate\n'
            'available models are `117M`, `345M`, `774M` or `1558M`\n'
            'Get current state by `!getconfig`.')

    @commands.command()
    @commands.guild_only()
    @commands.has_permissions(manage_messages=True)
    async def setconfig(self, ctx, nsamples: int, length: int, temp: float, top_k: int, model_name: str):
        if (self.not_ready):
            await ctx.send(self.not_ready_s)
            return
        logging.info('SET CONFIGURATION.')
        if (self.is_interfering):
            await ctx.send('Currently talking to someone. Try again later.')
            return
        if (self.not_ready):
            await ctx.send(self.not_ready_s)
            return
        if model_name not in self.models:
            await ctx.send('Model ' + model_name+ ' does not exist. Please choose a different model!')
            return
        server_id = ctx.message.guild.id

        await ctx.trigger_typing()
        if int(nsamples) * int(length) <= sizeLimit:
            await ctx.send('Setting configuration. Please wait...')
            self.serverSessions[server_id].shutdown()
            self.serverSessions[server_id].set_state(int(nsamples), int(length), float(temp), int(top_k), model_name)
            await ctx.send('**Using settings:**\n```'
                'N Samples: ' + str(nsamples) + "\n"
                'Max Length: ' + str(length) + "\n"
                'Temperature: ' + str(temp) + "\n"
                'Top K: ' + str(top_k) + "\n"
                'Model: ' + str(model_name) + "```")
            await ctx.trigger_typing()
            self.serverSessions[server_id].preinit_model()
            self.serverSessions[server_id].session = tf.Session()
            await ctx.trigger_typing()
            self.serverSessions[server_id].init_model()
            await ctx.send('Succesfully set configuration!')
            if (self.serverSessions[server_id].nsamples * self.serverSessions[server_id].length > 100):
                await ctx.send('The configuration parameters are process intensive, responses may take a while.')
        else:
            await ctx.send('Configuration failed. The configuration parameters too process intensive.')

    @commands.command()
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def debugsetconfig(self, ctx, nsamples: int, length: int, temp: float, top_k: int, model_name: str):
        if (self.not_ready):
            await ctx.send(self.not_ready_s)
            return
        logging.info('SET CONFIGURATION.')
        if (self.is_interfering):
            await ctx.send('Currently talking to someone. Try again later.')
            return
        if (self.not_ready):
            await ctx.send(self.not_ready_s)
            return
        if model_name not in self.models:
            await ctx.send('Model ' + model_name+ ' does not exist. Please choose a different model!')
            return
        server_id = ctx.message.guild.id

        await ctx.trigger_typing()
        await ctx.send('`CAUTION! Size limits are disabled. Please be considerate of everyone else who uses this. :)``')
        await ctx.send('`Setting configuration. Please wait...`')
        self.serverSessions[server_id].shutdown()
        await ctx.send('`Shutting down tensorflow model...`')
        self.serverSessions[server_id].set_state(int(nsamples), int(length), float(temp), int(top_k), model_name)
        await ctx.trigger_typing()
        await ctx.send('`Preinit tensorflow model...`')
        self.serverSessions[server_id].preinit_model()
        self.serverSessions[server_id].session = tf.Session()
        await ctx.trigger_typing()
        await ctx.send('`Setting up new tensorflow model...`')
        self.serverSessions[server_id].init_model()
        await ctx.send('`Succesfully set configuration!`')
        if (self.serverSessions[server_id].nsamples * self.serverSessions[server_id].length > 100):
            await ctx.send('`nsamples: ' + str(self.serverSessions[server_id].nsamples) + ' * length: ' + str(self.serverSessions[server_id].length) + ' '
            '(' + str(self.serverSessions[server_id].nsamples * self.serverSessions[server_id].length) + ') ' + 'is above the warning threshold of 100`')
            await ctx.send('`The configuration parameters are process intensive, responses may take a while...`')

    @commands.command()
    @commands.guild_only()
    @commands.has_permissions(manage_messages=True)
    async def default(self, ctx):
        if (self.not_ready):
            await ctx.send(self.not_ready_s)
            return
        logging.info('Setting to DEFAULT configuration.')
        if (self.is_interfering):
            await ctx.send('Currently talking to someone. Try again later.')
            return
        if (self.not_ready):
            await ctx.send(self.not_ready_s)
            return
        server_id = ctx.message.guild.id

        await ctx.trigger_typing()
        self.serverSessions[server_id].shutdown()
        self.serverSessions[server_id].set_state(1,200,1,0,'117M')
        await ctx.trigger_typing()
        self.serverSessions[server_id].preinit_model()
        self.serverSessions[server_id].session = tf.Session()
        await ctx.trigger_typing()
        self.serverSessions[server_id].init_model()

        await ctx.send('Succesfully set `default` configuration!')

    @default.error
    @helpconfig.error
    @setconfig.error
    @debugsetconfig.error
    @getconfig.error
    async def default_error(self, ctx, error):
        if isinstance(error, commands.MissingPermissions):
            text = "Sorry {}, you do not have permissions to do that!".format(ctx.message.author)
            await ctx.send(text)
    @talk.error
    @debugtalk.error
    async def talk_error(self, ctx, error):
        if isinstance(error, commands.errors.CommandInvokeError):
            self.is_interfering=False
            logging.info(error.original)
            print(error.original)
            await ctx.send('Command failed!')
        if isinstance(error, commands.errors.MissingRequiredArgument):
            #text = "You must deliver a message to me nyan!"
            #await ctx.send(text)
            logging.info('MSG being generated!')
            if (self.is_interfering):
                await ctx.send('Currently talking to someone. Try again later.')
                return
            if (self.not_ready):
                await ctx.send(self.not_ready_s)
                return
            server_id = ctx.message.guild.id
            logging.info('Guild: ' + str(server_id))
            self.is_interfering = True
            for _ in range(self.serverSessions[server_id].nsamples):
                async with ctx.typing():
                    start = time.time()
                    text_generator = functools.partial(self.serverSessions[server_id].generate_uncon_text)
                    out = await self.bot.loop.run_in_executor(None, text_generator)
                    response = self.serverSessions[server_id].enc.decode(out[0])
                    logging.info('RESPONSE GENERATED IN :' + str(round(time.time() - start, 2)) + ' seconds.')
                    logging.info('RESPONSE: ' + response)
                    logging.info('RESPONSE LEN: ' + str(len(response)))

                    response_chunk = 0
                    chunk_size = 1990
                    if (len(response) > 2000):
                        while (len(response) > response_chunk):
                            await ctx.send(response[response_chunk:response_chunk + chunk_size])
                            response_chunk += chunk_size
                    else:
                        await ctx.send(response)

            self.is_interfering = False

    @commands.Cog.listener()
    async def on_guild_join(self, guild):
        logging.info('Joined Guild.')
        self.serverSessions[guild.id] = gpt2_server_sessions(guild.id)
        logging.info('Spawned GPT-2 for new guild')

    @commands.Cog.listener()
    async def on_guild_remove(self, guild):
        logging.info('Removed from Guild.')
        self.serverSessions[guild.id].shutdown()
        logging.info('Despawned GPT-2 for said guild')

    @commands.Cog.listener()
    async def on_ready(self):
        if self.not_ready:
            self.not_ready = False
            guilds = await self.bot.fetch_guilds(limit=150).flatten()
            for guild in guilds:
                self.guildIdList.append(guild.id)
            self.serverSessions = {}
            for serverid in self.guildIdList:
                self.serverSessions[serverid] = gpt2_server_sessions(serverid)
                logging.info('Spawned GPT-2')
                self.not_ready = False

def setup(bot):
    bot.add_cog(GPT2Bot(bot))
