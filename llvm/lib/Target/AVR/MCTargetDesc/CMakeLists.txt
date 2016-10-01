add_llvm_library(LLVMAVRDesc
  AVRELFObjectWriter.cpp
  AVRELFStreamer.cpp
  AVRMCAsmInfo.cpp
  AVRMCExpr.cpp
  AVRMCTargetDesc.cpp
  AVRTargetStreamer.cpp
)

add_dependencies(LLVMAVRDesc AVRCommonTableGen)

