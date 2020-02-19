import lldb
import side_effect

class Resolver:
  """This resolver class is just so I can read out the extra_args."""
  
  def __init__(self, bkpt, extra_args, dict):
      self.bkpt = bkpt
      side_effect.g_extra_args = extra_args
    
  def __callback__(self, sym_ctx):
      """Doesn't actually do anything."""
      return

  def get_short_help(self):
      return "I am a python breakpoint resolver that does nothing"

