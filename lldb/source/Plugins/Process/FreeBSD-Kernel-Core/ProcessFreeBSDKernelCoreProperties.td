include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "processfreebsdkernelcore", Path = "plugin.process.freebsd-kernel-core" in {
  def ReadOnly: Property<"read-only", "Boolean">,
    Global,
    DefaultTrue,
    Desc<"Disable memory writes to the core. This is enabled by default for safety reasons.">;
}
