import lldb


class FooSyntheticProvider:

    def __init__(self, valobj, dict):
        self.valobj = valobj
        self.update()

    def update(self):
        self.adjust_for_architecture()

    def num_children(self):
        return 1

    def get_child_at_index(self, index):
        if index != 0:
            return None
        return self.i_ptr.Dereference()

    def get_child_index(self, name):
        if name == "*i_ptr":
            return 0
        return None

    def adjust_for_architecture(self):
        self.lp64 = (
            self.valobj.GetTarget().GetProcess().GetAddressByteSize() == 8)
        self.is_little = (self.valobj.GetTarget().GetProcess(
        ).GetByteOrder() == lldb.eByteOrderLittle)
        self.pointer_size = self.valobj.GetTarget().GetProcess().GetAddressByteSize()
        self.bar = self.valobj.GetChildMemberWithName('b')
        self.i_ptr = self.bar.GetChildMemberWithName('i_ptr')
