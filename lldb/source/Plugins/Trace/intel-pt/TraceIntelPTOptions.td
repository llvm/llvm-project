include "../../../../source/Commands/OptionsBase.td"

// The information of the start commands here should match the description of
// the intel-pt section of the jLLDBTraceStart packet in the
// lldb/docs/lldb-gdb-remote.txt documentation file. Similarly, it should match
// the API help message of TraceIntelPT::GetStartConfigurationHelp().

let Command = "thread trace start intel pt" in {
  def thread_trace_start_intel_pt_size: Option<"size", "s">,
    Group<1>,
    Arg<"Value">,
    Desc<"Trace size in bytes per thread. It must be a power of 2 greater "
         "than or equal to 4096 (2^12). The trace is circular keeping "
         "the most recent data. Defaults to 4096 bytes.">;
  def thread_trace_start_intel_pt_tsc: Option<"tsc", "t">,
    Group<1>,
    Desc<"Enable the use of TSC timestamps. This is supported on all devices "
         "that support intel-pt.">;
  def thread_trace_start_intel_pt_psb_period: Option<"psb-period", "p">,
    Group<1>,
    Arg<"Value">,
    Desc<"This value defines the period in which PSB packets will be "
         "generated. A PSB packet is a synchronization packet that contains a "
         "TSC timestamp and the current absolute instruction pointer. "
         "This parameter can only be used if "
         "/sys/bus/event_source/devices/intel_pt/caps/psb_cyc is 1. Otherwise, "
         "the PSB period will be defined by the processor. If supported, valid "
         "values for this period can be found in "
         "/sys/bus/event_source/devices/intel_pt/caps/psb_periods which "
         "contains a hexadecimal number, whose bits represent valid values "
         "e.g. if bit 2 is set, then value 2 is valid. The psb_period value is "
         "converted to the approximate number of raw trace bytes between PSB "
         "packets as: 2 ^ (value + 11), e.g. value 3 means 16KiB between PSB "
         "packets. Defaults to 0 if supported.">;
}

let Command = "process trace start intel pt" in {
  def process_trace_start_intel_pt_thread_size: Option<"thread-size", "s">,
    Group<1>,
    Arg<"Value">,
    Desc<"Trace size in bytes per thread. It must be a power of 2 greater "
         "than or equal to 4096 (2^12). The trace is circular keeping "
         "the most recent data. Defaults to 4096 bytes.">;
  def process_trace_start_intel_pt_process_size_limit: Option<"total-size-limit", "l">,
    Group<1>,
    Arg<"Value">,
    Desc<"Maximum total trace size per process in bytes. This limit applies to "
         "the sum of the sizes of all thread traces of this process, excluding "
         "the ones created with the \"thread trace start\" command. "
         "Whenever a thread is attempted to be traced due to this command and "
         "the limit would be reached, the process is stopped with a "
         "\"processor trace\" reason, so that the user can retrace the process "
         "if needed. Defaults to 500MB.">;
  def process_trace_start_intel_pt_tsc: Option<"tsc", "t">,
    Group<1>,
    Desc<"Enable the use of TSC timestamps. This is supported on all devices "
         "that support intel-pt.">;
  def process_trace_start_intel_pt_psb_period: Option<"psb-period", "p">,
    Group<1>,
    Arg<"Value">,
    Desc<"This value defines the period in which PSB packets will be "
         "generated. A PSB packet is a synchronization packet that contains a "
         "TSC timestamp and the current absolute instruction pointer. "
         "This parameter can only be used if "
         "/sys/bus/event_source/devices/intel_pt/caps/psb_cyc is 1. Otherwise, "
         "the PSB period will be defined by the processor. If supported, valid "
         "values for this period can be found in "
         "/sys/bus/event_source/devices/intel_pt/caps/psb_periods which "
         "contains a hexadecimal number, whose bits represent valid values "
         "e.g. if bit 2 is set, then value 2 is valid. The psb_period value is "
         "converted to the approximate number of raw trace bytes between PSB "
         "packets as: 2 ^ (value + 11), e.g. value 3 means 16KiB between PSB "
         "packets. Defaults to 0 if supported.">;
}
