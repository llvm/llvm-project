add_llvm_library(LLVMBinaryFormat
  AMDGPUMetadataVerifier.cpp
  Dwarf.cpp
  Magic.cpp
  MsgPackReader.cpp
  MsgPackTypes.cpp
  MsgPackWriter.cpp
  Wasm.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/BinaryFormat
  )
  
