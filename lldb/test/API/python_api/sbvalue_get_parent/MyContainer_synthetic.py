import lldb


@lldb.synthetic("MyContainer")
class MyContainerSynthetic:
    """Formatter that mimicks libstdc++ std::vector use of CreateChildAtOffset."""

    valobj: lldb.SBValue

    def __init__(self, valobj: lldb.SBValue, _) -> None:
        self.valobj = valobj

    def update(self) -> bool:
        self.data = self.valobj.GetChildMemberWithName("data")
        self.element_type = self.data.GetType().GetPointeeType()
        self.element_size = self.element_type.GetByteSize()
        self.count = self.valobj.GetChildMemberWithName("count").GetValueAsUnsigned(0)
        return True

    def num_children(self) -> int:
        return self.count

    def get_child_at_index(self, index: int) -> lldb.SBValue:
        if 0 <= index < self.count:
            offset = index * self.element_size
            return self.data.CreateChildAtOffset(
                f"[{index}]", offset, self.element_type
            )
        return lldb.SBValue()
