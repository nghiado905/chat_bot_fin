from flask import *
from chatbot import *
app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/asking')
def ask():
    return render_template('asking.html')

@app.route('/answer')
def answer():
    query = request.args.get('input',default='',type=str)
    
    result = q_a.invoke(query)['result']
    print(result)
    return render_template('result.html', answer=result)




if __name__ == '__main__':
    db = read_vectors()
    template = create_template()
    prompt = set_custom_prompt(template)
    q_a = create_chain(db,prompt)
    app.run(debug=True)