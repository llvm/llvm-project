import lldb


class fooSynthProvider:

    def __init__(self, valobj, dict):
        self.valobj = valobj
        self.int_type = valobj.GetType().GetBasicType(lldb.eBasicTypeInt)

    def num_children(self):
        return 3

    def get_child_at_index(self, index):
        if index == 0:
            child = self.valobj.GetChildMemberWithName('a')
        if index == 1:
            child = self.valobj.CreateChildAtOffset('fake_a', 1, self.int_type)
        if index == 2:
            child = self.valobj.GetChildMemberWithName('r')
        return child

    def get_child_index(self, name):
        if name == 'a':
            return 0
        if name == 'fake_a':
            return 1
        return 2
