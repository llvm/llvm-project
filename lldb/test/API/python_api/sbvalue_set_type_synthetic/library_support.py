import lldb


@lldb.synthetic("SessionInfo")
class SessionInfoSynthetic:
    """Type synthetic with additional information about the underlying object that isn't represented in the debug info"""

    valobj: lldb.SBValue
    foos: lldb.SBValue
    bars: lldb.SBValue

    def __init__(self, valobj: lldb.SBValue, _) -> None:
        self.valobj = valobj

    def update(self) -> bool:
        self.foos = self.valobj.GetChildMemberWithName("foos")
        self.foos.SetTypeSynthetic(
            lldb.SBTypeSynthetic.CreateWithClassName(
                "library_support.FooHandleArraySynthetic"
            )
        )

        self.bars = self.valobj.GetChildMemberWithName("bars")
        self.bars.SetTypeSynthetic(
            lldb.SBTypeSynthetic.CreateWithClassName(
                "library_support.BarHandleArraySynthetic"
            )
        )
        return True

    def num_children(self) -> int:
        return 2

    def get_child_at_index(self, index: int) -> lldb.SBValue:
        if index == 0:
            return self.foos
        if index == 1:
            return self.bars

    def get_child_index(self, name: str) -> int:
        if name == "foos":
            return 0
        if name == "bars":
            return 1


class HandleArraySyntheticBase:
    valobj: lldb.SBValue
    valtype: lldb.SBType
    array: lldb.SBValue

    def __init__(self, valobj: lldb.SBValue, valtype: lldb.SBType) -> None:
        self.valobj = valobj
        self.valtype = valtype

        self.size = 0
        self.array = lldb.SBValue()

    def update(self) -> bool:
        self.size = self.valobj.GetChildMemberWithName("size").GetValueAsUnsigned(0)

        array_t = self.valtype.GetPointerType().GetArrayType(self.size).GetPointerType()
        self.array = self.valobj.GetChildMemberWithName("data").Cast(array_t)
        return True

    def num_children(self) -> int:
        return 2

    def get_child_at_index(self, index: int) -> lldb.SBValue:
        if index == 0:
            return self.valobj.GetChildMemberWithName("size")
        if index == 1:
            return self.array

    def get_child_index(self, name: str) -> int:
        if name == "size":
            return 0
        if name == "data":
            return 1


class FooHandleArraySynthetic(HandleArraySyntheticBase):
    def __init__(self, valobj: lldb.SBValue, _) -> None:
        super().__init__(valobj, valobj.target.FindFirstType("Foo"))


class BarHandleArraySynthetic(HandleArraySyntheticBase):
    def __init__(self, valobj: lldb.SBValue, _) -> None:
        super().__init__(valobj, valobj.target.FindFirstType("Bar"))
