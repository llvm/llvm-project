add_llvm_library(LLVMAVRDesc
  AVRAsmBackend.cpp
  AVRELFObjectWriter.cpp
  AVRELFStreamer.cpp
  AVRMCAsmInfo.cpp
  AVRMCCodeEmitter.cpp
  AVRMCELFStreamer.cpp
  AVRMCExpr.cpp
  AVRMCTargetDesc.cpp
  AVRTargetStreamer.cpp
)

add_dependencies(LLVMAVRDesc AVRCommonTableGen)

