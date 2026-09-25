import lldb


def _summary(valobj: lldb.SBValue) -> str:
    size = (
        valobj.GetNonSyntheticValue()
        .GetChildMemberWithName("size")
        .GetValueAsUnsigned()
    )
    return f"size={size}"


@lldb.summary("IntContainer", expand=True)
def IntContainerSummary(valobj: lldb.SBValue, _):
    return _summary(valobj)


@lldb.summary("^Container<.+>$", regex=True, expand=True)
def ContainerSummary(valobj, _):
    return _summary(valobj)


class _ContainerSyntheticBase:
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


@lldb.synthetic("IntContainer")
class IntContainerSynthetic(_ContainerSyntheticBase):
    pass


@lldb.synthetic("^Container<.+>$", regex=True)
class ContainerSynthetic(_ContainerSyntheticBase):
    pass
