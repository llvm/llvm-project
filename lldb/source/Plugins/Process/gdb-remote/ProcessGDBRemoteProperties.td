include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "processgdbremote" in {
  def PacketTimeout: Property<"packet-timeout", "UInt64">,
    Global,
    DefaultUnsignedValue<5>,
    Desc<"Specify the default packet timeout in seconds.">;
  def TargetDefinitionFile: Property<"target-definition-file", "FileSpec">,
    Global,
    DefaultStringValue<"">,
    Desc<"The file that provides the description for remote target registers.">;
  def UseSVR4: Property<"use-libraries-svr4", "Boolean">,
    Global,
    DefaultFalse,
    Desc<"If true, the libraries-svr4 feature will be used to get a hold of the process's loaded modules.">;
  def UseGPacketForReading: Property<"use-g-packet-for-reading", "Boolean">,
    Global,
    DefaultFalse,
    Desc<"Specify if the server should use 'g' packets to read registers.">;
}
