include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "platformwasm" in {
  def RuntimePath
      : Property<"runtime-path", "FileSpec">,
        Global,
        DefaultStringValue<"">,
        Desc<"Path to the WebAssembly runtime binary. If the path does not "
             "contain a directory separator, the filename is looked up in the "
             "PATH environment variable. If empty, the filename is derived "
             "from the architecture setting.">;
  def RuntimeArgs : Property<"runtime-args", "Args">,
                    Global,
                    DefaultStringValue<"">,
                    Desc<"Extra arguments to pass to the WebAssembly runtime.">;
  def PortArg : Property<"port-arg", "String">,
                Global,
                DefaultStringValue<"">,
                Desc<"Argument to the WebAssembly runtime to specify the "
                     "GDB remote port.">;
}
