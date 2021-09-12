add_llvm_component_library(LLVMOrcTargetProcess
  JITLoaderGDB.cpp
  RegisterEHFrames.cpp
  SimpleRemoteEPCServer.cpp
  TargetExecutionUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/Orc

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}

  LINK_COMPONENTS
  OrcShared
  Support
  )
