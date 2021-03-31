include "../../../../source/Commands/OptionsBase.td"

let Command = "thread trace start intel pt" in {
  def thread_trace_start_intel_pt_size: Option<"size", "s">,
    Group<1>,
    Arg<"Value">,
    Desc<"Trace size in bytes per thread. It must be a power of 2 greater "
         "than or equal to 4096 (2^12). The trace is circular keeping "
         "the most recent data. Defaults to 4096 bytes.">;
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
         "the limit would be reached, the process is stopped with a \"tracing\" "
         "reason, so that the user can retrace the process if needed.">;
}
