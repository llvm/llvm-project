import lldb


@lldb.synthetic("Foo")
class FooSynthetic:
    """Dummy synthetic to demonstrate override capabilities"""

    def __init__(self, valobj: lldb.SBValue, _) -> None:
        pass


class BarSynthetic:
    """Dummy synthetic to demonstrate override capabilities"""

    def __init__(self, valobj: lldb.SBValue, _) -> None:
        pass
