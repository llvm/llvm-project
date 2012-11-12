import lldb
class jasSynthProvider:
     def __init__(self, valobj, dict):
         self.valobj = valobj;
     def num_children(self):
         return 2;
     def get_child_at_index(self, index):
         child = None
         if index == 0:
             child = self.valobj.GetChildMemberWithName('A');
         if index == 1:
             child = self.valobj.CreateValueFromExpression('X', '(int)1')
         return child;
     def get_child_index(self, name):
         if name == 'A':
             return 0;
         if name == 'X':
             return 1;
         return None;

def __lldb_init_module(debugger,dict):
     debugger.CreateCategory("JASSynth").AddTypeSynthetic(lldb.SBTypeNameSpecifier("JustAStruct"),
        lldb.SBTypeSynthetic.CreateWithClassName("jas_synth.jasSynthProvider"))

