add_llvm_component_library(LLVMOrcTargetProcess
  JITLoaderGDB.cpp
  RegisterEHFrames.cpp
  TargetExecutionUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/Orc

  LINK_COMPONENTS
  OrcShared
  Support
  )
