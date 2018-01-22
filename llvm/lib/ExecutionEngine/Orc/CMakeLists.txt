add_llvm_library(LLVMOrcJIT
  Core.cpp
  ExecutionUtils.cpp
  IndirectionUtils.cpp
  Legacy.cpp
  NullResolver.cpp
  OrcABISupport.cpp
  OrcCBindings.cpp
  OrcError.cpp
  OrcMCJITReplacement.cpp
  RPCUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/Orc

  DEPENDS
  intrinsics_gen
  )
