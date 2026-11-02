include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "platformwasm", Path = "platform.plugin.wasm" in {
  def RuntimePath : Property<"runtime-path", "FileSpec">,
                    Global,
                    DefaultStringValue<"">,
                    Desc<"Path to the WebAssembly runtime binary. If the path "
                         "does not contain a directory separator, the filename "
                         "is looked up in the PATH environment variable.">;
  def PortArg : Property<"port-arg", "String">,
                Global,
                DefaultStringValue<"">,
                Desc<"Argument to the WebAssembly runtime to specify the "
                     "GDB remote port. The port number chosen by LLDB will be "
                     "concatenated to this argument, which therefore has to "
                     "carry its value in the same argument. For example: "
                     "`-g=127.0.0.1:` or `--debugger-port=`.">;
  def EnvArg : Property<"env-arg", "String">,
               Global,
               DefaultStringValue<"">,
               Desc<"Prefix of the argument that forwards an environment "
                    "variable to the WebAssembly runtime. LLDB does not read "
                    "the variables from this setting. It takes them from the "
                    "launch environment, such as `target.env-vars` or "
                    "`process launch -E`, and appends each `key=value` pair "
                    "to this prefix, one argument per variable. For example, "
                    "`--env=` yields `--env=A=1 --env=B=2`. The host "
                    "environment is not forwarded. If empty, no environment "
                    "is forwarded.">;
  def RuntimeArgs : Property<"runtime-args", "Args">,
                    Global,
                    DefaultStringValue<"">,
                    Desc<"Extra arguments to pass to the WebAssembly runtime. "
                         "They precede the port argument, so a runtime that "
                         "dispatches on a leading subcommand names it here. "
                         "For the argument that specifies the GDB remote port, "
                         "use port-arg instead.">;
}
