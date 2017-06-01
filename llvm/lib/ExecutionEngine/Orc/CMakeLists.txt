add_llvm_library(LLVMOrcJIT
  ExecutionUtils.cpp
  IndirectionUtils.cpp
  NullResolver.cpp
  OrcABISupport.cpp
  OrcCBindings.cpp
  OrcError.cpp
  OrcMCJITReplacement.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/Orc

  DEPENDS
  intrinsics_gen
  )
