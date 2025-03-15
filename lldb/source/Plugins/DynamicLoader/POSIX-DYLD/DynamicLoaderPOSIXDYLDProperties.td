
include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "dynamicloaderposixdyld" in {
  def ParallelModuleLoad: Property<"parallel-module-load", "Boolean">,
    DefaultFalse,
    Desc<"Enable experimental loading of modules in parallel for the POSIX dynamic loader.">;
}
