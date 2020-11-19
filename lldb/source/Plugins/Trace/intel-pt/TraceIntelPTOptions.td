include "../../../../source/Commands/OptionsBase.td"

let Command = "thread trace start intel pt" in {
  def thread_trace_start_intel_pt_size: Option<"size", "s">,
    Group<1>,
    Arg<"Value">,
    Desc<"The size of the trace in KB. The kernel rounds it down to the nearest"
         " multiple of 4. Defaults to 4.">;
  def thread_trace_start_intel_pt_custom_config: Option<"custom-config", "c">,
    Group<1>,
    Arg<"Value">,
    Desc<"Low level bitmask configuration for the kernel based on the values "
         "in `grep -H  /sys/bus/event_source/devices/intel_pt/format/*`. "
         "See https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf-intel-pt.txt"
         " for more information. Defaults to 0.">;
}
