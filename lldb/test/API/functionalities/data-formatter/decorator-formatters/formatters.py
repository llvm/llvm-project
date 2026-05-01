import lldb


@lldb.summary("Pair")
def pair_summary(valobj, _):
    first = valobj.GetChildMemberWithName("first").GetValueAsSigned()
    second = valobj.GetChildMemberWithName("second").GetValueAsSigned()
    return f"({first}, {second})"


@lldb.synthetic("Container")
class ContainerSyntheticProvider:
    valobj: lldb.SBValue
    count: int
    items: lldb.SBValue

    def __init__(self, valobj: lldb.SBValue, _) -> None:
        self.valobj = valobj

    def update(self) -> bool:
        self.count = self.valobj.GetChildMemberWithName("size").GetValueAsSigned()
        self.items = self.valobj.GetChildMemberWithName("items")
        return True

    def num_children(self) -> int:
        return self.count

    def get_child_at_index(self, index: int) -> lldb.SBValue:
        return self.items.GetChildAtIndex(index)
