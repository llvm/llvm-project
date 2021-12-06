add_llvm_component_library(LLVMDebuginfod
  Debuginfod.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Debuginfod

  LINK_COMPONENTS
  Support
  )
