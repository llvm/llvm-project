add_llvm_library(LLVMDebugInfoMSF
  MappedBlockStream.cpp
  MSFBuilder.cpp
  MSFCommon.cpp
  MSFError.cpp
  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/MSF"
  )
