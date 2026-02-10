include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "processkdp", Parent = "plugin.process.kdp" in {
  def KDPPacketTimeout: Property<"packet-timeout", "UInt64">,
    Global,
    DefaultUnsignedValue<5>,
    Desc<"Specify the default packet timeout in seconds.">;
}
