include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "dynamicloaderdarwin_experimental" in {
  def EnableParallelImageLoad: Property<"enable-parallel-image-load", "Boolean">,
    Global,
    DefaultTrue,
    Desc<"Load images in parallel.">;
}
