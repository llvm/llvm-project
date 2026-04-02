add_llvm_component_library(LLVMNVPTXDesc
  NVPTXInstPrinter.cpp
  NVPTXMCAsmInfo.cpp
  NVPTXMCTargetDesc.cpp
  NVPTXTargetStreamer.cpp

  LINK_COMPONENTS
  MC
  NVPTXInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  NVPTX
  )

# Static table and helper function redefinitions.
set_source_files_properties(
  NVPTXMCTargetDesc.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
