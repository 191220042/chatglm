
import click
import gradio as gr
from transformers.utils.versions import require_version
import os




from load_model import myWebChatModel
from my_chatbot import create_chat_box
from glmtuner.tuner import get_infer_args
from glmtuner.webui.manager import Manager


require_version("gradio>=3.36.0", "To fix: pip install gradio>=3.36.0")

@click.command
@click.option("--model_name_or_path", default= "THUDM/chatglm2-6b", help="The name or the path of the model.")
@click.option("--port",default=7984, help="The port of the server.")
def main(model_name_or_path,port):
    my_model = myWebChatModel(*get_infer_args())

    with gr.Blocks(title="Web Demo") as demo:
        lang = gr.Dropdown(choices=["en", "zh"], value="zh")

        _, _, _, chat_elems = create_chat_box(my_model, visible=True)

        manager = Manager([{"lang": lang}, chat_elems])

        demo.load(manager.gen_label, [lang], [lang] + list(chat_elems.values()))

        lang.change(manager.gen_label, [lang], [lang] + list(chat_elems.values()))

    os.environ['no_proxy']='127.0.0.1,localhost'
    demo.queue()
    demo.launch(server_name="localhost", server_port=port, share=False, inbrowser=True)



if __name__ == "__main__":
    main()
