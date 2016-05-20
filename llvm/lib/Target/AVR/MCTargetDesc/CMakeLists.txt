add_llvm_library(LLVMAVRDesc
  AVRTargetStreamer.cpp
  AVRELFStreamer.cpp
)

add_dependencies(LLVMAVRDesc AVRCommonTableGen)

