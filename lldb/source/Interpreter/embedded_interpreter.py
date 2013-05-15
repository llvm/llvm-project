import readline
import code
import sys
import traceback

class SimpleREPL(code.InteractiveConsole):
   def __init__(self, prompt, dict):
       code.InteractiveConsole.__init__(self,dict)
       self.prompt = prompt
       self.loop_exit = False
       self.dict = dict

   def interact(self):
       try:
           sys.ps1
       except AttributeError:
           sys.ps1 = ">>> "
       try:
           sys.ps2
       except AttributeError:
           sys.ps2 = "... "

       while not self.loop_exit:
           try:
               self.read_py_command()
           except (SystemExit, EOFError):
               # EOF while in Python just breaks out to top level.
               self.write('\n')
               self.loop_exit = True
               break
           except KeyboardInterrupt:
               self.write("\nKeyboardInterrupt\n")
               self.resetbuffer()
               more = 0
           except:
               traceback.print_exc()

   def process_input (self, in_str):
      # Canonicalize the format of the input string
      temp_str = in_str
      temp_str.strip(' \t')
      words = temp_str.split()
      temp_str = ('').join(words)

      # Check the input string to see if it was the quit
      # command.  If so, intercept it, so that it doesn't
      # close stdin on us!
      if (temp_str.lower() == "quit()" or temp_str.lower() == "exit()"):
         self.loop_exit = True
         in_str = "raise SystemExit "
      return in_str

   def my_raw_input (self, prompt):
      stream = sys.stdout
      stream.write (prompt)
      stream.flush ()
      try:
         line = sys.stdin.readline()
      except KeyboardInterrupt:
         line = " \n"
      except (SystemExit, EOFError):
         line = "quit()\n"
      if not line:
         raise EOFError
      if line[-1] == '\n':
         line = line[:-1]
      return line

   def read_py_command(self):
       # Read off a complete Python command.
       more = 0
       while 1:
           if more:
               prompt = sys.ps2
           else:
               prompt = sys.ps1
           line = self.my_raw_input(prompt)
           # Can be None if sys.stdin was redefined
           encoding = getattr(sys.stdin, "encoding", None)
           if encoding and not isinstance(line, unicode):
               line = line.decode(encoding)
           line = self.process_input (line)
           more = self.push(line)
           if not more:
               break

   def one_line (self, input):
      line = self.process_input (input)
      more = self.push(line)
      if more:
         self.write ("Input not a complete line.")
         self.resetbuffer()
         more = 0

def run_python_interpreter (dict):
   # Pass in the dictionary, for continuity from one session to the next.
   repl = SimpleREPL('>>> ', dict)
   repl.interact()

def run_one_line (dict, input_string):
   repl = SimpleREPL ('', dict)
   repl.one_line (input_string)

