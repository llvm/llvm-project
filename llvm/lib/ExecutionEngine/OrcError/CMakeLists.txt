add_llvm_component_library(LLVMOrcError
  OrcError.cpp
  RPCError.cpp
  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/Orc
  )
