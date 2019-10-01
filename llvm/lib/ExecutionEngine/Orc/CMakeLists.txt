add_llvm_library(LLVMOrcJIT
  CompileOnDemandLayer.cpp
  CompileUtils.cpp
  Core.cpp
  ExecutionUtils.cpp
  IndirectionUtils.cpp
  IRCompileLayer.cpp
  IRTransformLayer.cpp
  JITTargetMachineBuilder.cpp
  LazyReexports.cpp
  Legacy.cpp
  Layer.cpp
  LLJIT.cpp
  NullResolver.cpp
  ObjectLinkingLayer.cpp
  ObjectTransformLayer.cpp
  OrcABISupport.cpp
  OrcCBindings.cpp
  OrcError.cpp
  OrcMCJITReplacement.cpp
  RPCUtils.cpp
  RTDyldObjectLinkingLayer.cpp
  ThreadSafeModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/Orc

  DEPENDS
  intrinsics_gen
  )

target_link_libraries(LLVMOrcJIT
  PRIVATE
  LLVMBitReader
  LLVMBitWriter
  )
