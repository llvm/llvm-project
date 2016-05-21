add_llvm_library(LLVMAVRDesc
  AVRELFStreamer.cpp
  AVRMCAsmInfo.cpp
  AVRTargetStreamer.cpp
)

add_dependencies(LLVMAVRDesc AVRCommonTableGen)

