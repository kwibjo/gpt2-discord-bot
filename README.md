# GPT-2 Discord Bot

### Setup

- Install dependencies with:

```bash
pip install -r requirements.txt
```

- Run the script `download_model.sh` by:
```
python download_model.py 1558M
```
_This should download the gpt-2 model. `117M` is the smallest model, `345M, 774M` are larger and `1558M` is the largest variant._

- Create `config` folder

- Create `servers` folder in `config` folder

- Create `auth.json`, and place it inside the `config` folder. Its content should be:

```json
{
   "token": "<your_token>",
   "client_id": "<client_id>"
}
```

### How to run

- Run the script with:

```bash
python3 gpt-chatbot-client.py
```

- use it!

```use
!talk Complete this sentence
!talk (No text here to generate unconditional sample)
```

### Commands/Settings
Each server gets its own Tensorflow session with its own model. This gives every server the opportunity to use it's own GPT-2 model.  
The !setconfig command sets the neccessary parameters!  
Only user with message managing permissions on the respective servers can user the following commands:
```conf_server
!setconfig <nsamples> <length> <temperature> <topk> <model: 117M, 345M, 774M or 1558M>
!getconfig
!default
```
!default resets the settings for the server to the default settings nsamples=1, length=200, temperature=1, top_k=0, model=117M

### Improvements

- Enable finetuning.
