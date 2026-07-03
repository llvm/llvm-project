import lldb


class wrapPtrSynthProvider:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def num_children(self):
        return 1

    def get_child_at_index(self, index):
        if index == 0:
            return self.valobj.GetChildMemberWithName("ptr")
        if index == 1:
            internal_child = self.valobj.GetChildMemberWithName("ptr")
            if not internal_child:
                return None
            return internal_child.Dereference()
        return None

    def get_child_index(self, name):
        if name == "ptr":
            return 0
        if name == "$$dereference$$":
            return 1
        return -1

    def update(self):
        return True
