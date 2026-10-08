include "../../include/lldb/Core/EnumsBase.td"

def watch_type : EnumDef<"OptionGroupWatchpoint::", [
  EnumValue<"eWatchRead", "read", "Watch for read">,
  EnumValue<"eWatchWrite", "write", "Watch for write">,
  EnumValue<"eWatchModify", "modify", "Watch for modifications">,
  EnumValue<"eWatchReadWrite", "read_write", "Watch for read/write">,
]>;
