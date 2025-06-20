import lldb

class smartPtrSynthProvider:
    def __init__(self, valobj, dict):
        self.valobj = valobj

    def num_children(self):
        return 1

    def get_child_at_index(self, index):
        if index == 0:
            return self.valobj.GetChildMemberWithName("__ptr_")
        if index == 1:
            internal_child = self.valobj.GetChildMemberWithName("__ptr_")
            if not internal_child:
                return None
            value_type = internal_child.GetType().GetPointerType()
            cast_ptr_sp = internal_child.Cast(value_type)
            value = internal_child.Dereference()
            return value
        return None

    def get_child_index(self, name):
        if name == "__ptr_":
            return 0
        if name == "$$dereference$$":
            return 1
        return -1

    def update(self):
        return True
