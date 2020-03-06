import lldb

counter = 0


class ftsp:

    def __init__(self, valobj, dict):
        self.valobj = valobj

    def num_children(self):
        if self.char.IsValid():
            return 5
        return 0

    def get_child_index(self, name):
        return 0

    def get_child_at_index(self, index):
        if index == 0:
            return self.x.Cast(self.char)
        if index == 4:
            return self.valobj.CreateValueFromExpression(
                str(index), '(char)(' + str(self.count) + ')')
        return self.x.CreateChildAtOffset(str(index),
                                          index,
                                          self.char)

    def update(self):
        self.x = self.valobj.GetChildMemberWithName('x')
        self.char = self.valobj.GetType().GetBasicType(lldb.eBasicTypeChar)
        global counter
        self.count = counter
        counter = counter + 1
        return True  # important: if we return False here, or fail to return, the test will fail


def __lldb_init_module(debugger, dict):
    global counter
    counter = 0
