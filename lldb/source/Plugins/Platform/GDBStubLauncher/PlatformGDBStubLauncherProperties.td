include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "platformgdbstublauncher",
    Path = "platform.plugin.gdb-stub-launcher" in {
  def LaunchCommand : Property<"launch-command", "String">,
                      Global,
                      DefaultStringValue<"">,
                      Desc<"The command `process launch` runs, via `/bin/sh "
                           "-c`, to start a program that serves a gdb-remote "
                           "stub. {port} is replaced with a free TCP port and "
                           "{executable} with the target's executable, shell "
                           "quoted. Required. A command with options is set "
                           "with `settings set -- <name> <command>`. One target "
                           "at a time can have a program running.">;
  def WorkingDir : Property<"working-dir", "FileSpec">,
                   Global,
                   DefaultStringValue<"">,
                   Desc<"The directory launch-command runs in.">;
  def PortEnvVar : Property<"port-env-var", "String">,
                   Global,
                   DefaultStringValue<"">,
                   Desc<"An environment variable to pass the chosen port in, "
                        "for a program that reads it from there rather than "
                        "from its command line.">;
  def Connect : Property<"connect", "Enum">,
                Global,
                DefaultEnumValue<"eConnectGDBStub">,
                EnumValues<"OptionEnumValues(g_connect_enums)">,
                Desc<"What to attach to once the program is up: its gdb stub "
                     "on the chosen port, or one process of a gdb-remote "
                     "platform server at connect-url.">;
  def ConnectURL : Property<"connect-url", "String">,
                   Global,
                   DefaultStringValue<"">,
                   Desc<"With connect=remote-platform, the URL of the platform "
                        "server, e.g. connect://127.0.0.1:4000.">;
  def Attach : Property<"attach", "String">,
               Global,
               DefaultStringValue<"">,
               Desc<"With connect=remote-platform, the process to attach to, "
                    "by name or pid.">;
  def ReadyLog : Property<"ready-log", "FileSpec">,
                 Global,
                 DefaultStringValue<"">,
                 Desc<"A log to wait on before connecting, until text written "
                      "to it after launch contains ready-pattern. A relative "
                      "path is under working-dir.">;
  def ReadyPattern : Property<"ready-pattern", "String">,
                     Global,
                     DefaultStringValue<"">,
                     Desc<"The text ready-log has to show before connecting.">;
  def RequireHardwareBreakpoints
      : Property<"require-hardware-breakpoints", "Boolean">,
        Global,
        DefaultFalse,
        Desc<"Set target.require-hardware-breakpoint on the target and make "
             "its existing breakpoints hardware, for a stub that accepts "
             "software breakpoints but never traps on them. It stays set.">;
  def Timeout : Property<"timeout", "UInt64">,
                Global,
                DefaultUnsignedValue<120>,
                Desc<"Seconds to wait for the program to serve before giving "
                     "up.">;
  def Log : Property<"log", "FileSpec">,
            Global,
            DefaultStringValue<"">,
            Desc<"File the program's stdout and stderr go to, truncated on "
                 "each launch. If empty, gdb-stub-launcher-<lldb pid>.log in "
                 "the temporary directory. The program runs off the terminal, "
                 "in its own process group.">;
  def Architecture : Property<"architecture", "String">,
                     Global,
                     DefaultStringValue<"">,
                     Desc<"Architecture to report as supported. If empty, the "
                          "host architecture is used.">;
}
