import lldb


@lldb.summary("Ignored", invalid=True)
def IgnoredSummary(valobj: lldb.SBValue, _) -> str:
    return "nope"
