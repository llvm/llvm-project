include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "platformwasm" in {
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
                     "concatenated to this argument. For example: "
                     "`-g=127.0.0.1:` or `--debugger-port `.">;
  def RuntimeArgs : Property<"runtime-args", "Args">,
                    Global,
                    DefaultStringValue<"">,
                    Desc<"Extra arguments to pass to the WebAssembly runtime. "
                         "For the argument that specifies the GDB remote port, "
                         "use port-arg instead.">;
}
