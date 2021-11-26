include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "platformqemuuser" in {
  def Architecture: Property<"architecture", "String">,
    Global,
    DefaultStringValue<"">,
    Desc<"Architecture to emulate.">;
  def EmulatorPath: Property<"emulator-path", "FileSpec">,
    Global,
    DefaultStringValue<"">,
    Desc<"Path to the emulator binary.">;
  def EmulatorArgs: Property<"emulator-args", "Args">,
    Global,
    DefaultStringValue<"">,
    Desc<"Extra arguments to pass to the emulator.">;
}
