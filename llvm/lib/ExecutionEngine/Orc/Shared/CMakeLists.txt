add_llvm_component_library(LLVMOrcShared
  OrcError.cpp
  RPCError.cpp
  TargetProcessControlTypes.cpp
  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/Orc

  LINK_COMPONENTS
  Support
  )
