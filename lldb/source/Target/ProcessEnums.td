include "../../include/lldb/Core/EnumsBase.td"

def follow_fork_mode_values : EnumDef<"", [
    EnumValue<"eFollowParent", "parent",
      "Continue tracing the parent process and detach the child.">,
    EnumValue<"eFollowChild", "child",
      "Trace the child process and detach the parent.">,
  ]>;

