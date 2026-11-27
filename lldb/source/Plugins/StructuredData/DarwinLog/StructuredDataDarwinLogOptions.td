include "../../../../source/Commands/OptionsBase.td"

let Command = "plugin structured_data darwin_log enable" in {
  def darwin_log_enable_activity_chain
      : Option<"activity-chain", "C">,
        Desc<"Include the activity parent-child chain in the message header "
             "when displaying a log message.  The activity hierarchy is "
             "displayed as {grandparent-activity}:"
             "{parent-activity}:{activity}[:...].">;
  def darwin_log_enable_all_fields
      : Option<"all-fields", "A">,
        Desc<"Shortcut to specify that all header fields should be "
             "displayed.">;
  def darwin_log_enable_any_process
      : Option<"any-process", "a">,
        Desc<"Specifies log messages from other related processes should be "
             "included.">;
  def darwin_log_enable_broadcast_events
      : Option<"broadcast-events", "b">,
        Arg<"Boolean">,
        Desc<"Specify if the plugin should broadcast events.  Broadcasting "
             "log events is a requirement for displaying the log entries in "
             "LLDB command-line.  It is also required if LLDB clients want to "
             "process log events.  The default is true.">;
  def darwin_log_enable_category
      : Option<"category", "c">,
        Desc<"Include the category in the message header when displaying a "
             "log message.">;
  def darwin_log_enable_debug
      : Option<"debug", "d">,
        Desc<"Specifies debug-level log messages should be included.  "
             "Specifying --debug implies --info.">;
  def darwin_log_enable_echo_to_stderr
      : Option<"echo-to-stderr", "e">,
        Arg<"Boolean">,
        Desc<"Specify whether os_log()/NSLog() messages are echoed to the "
             "target program's stderr.  When DarwinLog is enabled, we shut off "
             "the mirroring of os_log()/NSLog() to the program's stderr.  "
             "Setting this flag to true will restore the stderr mirroring."
             "The default is false.">;
  def darwin_log_enable_filter
      : Option<"filter", "f">,
        Arg<"DarwinLogFilter">,
        // There doesn't appear to be a great way to have these multi-line,
        // formatted tables in help.  This looks mostly right but there are
        // extra linefeeds added at seemingly random spots, and indentation
        // isn't handled properly on those lines.
        Desc<"Appends a filter rule to the log message filter chain.  Multiple "
             "rules may be added by specifying this option multiple times, "
             "once per filter rule.  Filter rules are processed in the order "
             "they are specified, with the --no-match-accepts setting used "
             "for any message that doesn't match one of the rules.\n"
             "\n"
             "    Filter spec format:\n"
             "\n"
             "    --filter \"{action} {attribute} {op}\"\n"
             "\n"
             "    {action} :=\n"
             "      accept |\n"
             "      reject\n"
             "\n"
             "    {attribute} :=\n"
             "       activity       |  // message's most-derived activity\n"
             "       activity-chain |  // message's {parent}:{child} activity\n"
             "       category       |  // message's category\n"
             "       message        |  // message's expanded contents\n"
             "       subsystem      |  // message's subsystem\n"
             "\n"
             "    {op} :=\n"
             "      match {exact-match-text} |\n"
             "      regex {search-regex}\n"
             "\n"
             "The regex flavor used is the C++ std::regex ECMAScript format.  "
             "Prefer character classes like [[:digit:]] to \\d and the like, "
             "as getting the backslashes escaped through properly is "
             "error-prone.">;
  def darwin_log_enable_info
      : Option<"info", "i">,
        Desc<"Specifies info-level log messages should be included.">;
  def darwin_log_enable_live_stream
      : Option<"live-stream", "l">,
        Arg<"Boolean">,
        Desc<"Specify whether logging events are live-streamed or buffered.  "
             "True indicates live streaming, false indicates buffered.  The "
             "default is true (live streaming).  Live streaming will deliver "
             "log messages with less delay, but buffered capture mode has less "
             "of an observer effect.">;
  def darwin_log_enable_no_match_accepts
      : Option<"no-match-accepts", "n">,
        Arg<"Boolean">,
        Desc<"Specify whether a log message that doesn't match any filter rule "
             "is accepted or rejected, where true indicates accept.  The "
             "default is true.">;
  def darwin_log_enable_subsystem
      : Option<"subsystem", "s">,
        Desc<"Include the subsystem in the message header when displaying a "
             "log message.">;
  def darwin_log_enable_timestamp_relative
      : Option<"timestamp-relative", "r">,
        Desc<"Include timestamp in the message header when printing a log "
             "message.  The timestamp is relative to the first displayed "
             "message.">;
}
