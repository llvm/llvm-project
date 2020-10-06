include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "platformdarwinkernel" in {
  def KextDirectories: Property<"kext-directories", "FileSpecList">,
    DefaultStringValue<"">,
    Desc<"Directories/KDKs to search for kexts in when starting a kernel debug session.">;
}
