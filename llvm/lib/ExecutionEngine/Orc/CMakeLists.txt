add_llvm_library(LLVMOrcJIT
  Core.cpp
  ExecutionUtils.cpp
  IndirectionUtils.cpp
  IRCompileLayer.cpp
  IRTransformLayer.cpp
  Legacy.cpp
  Layer.cpp
  NullResolver.cpp
  ObjectTransformLayer.cpp
  OrcABISupport.cpp
  OrcCBindings.cpp
  OrcError.cpp
  OrcMCJITReplacement.cpp
  RPCUtils.cpp
  RTDyldObjectLinkingLayer.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/Orc

  DEPENDS
  intrinsics_gen
  )
