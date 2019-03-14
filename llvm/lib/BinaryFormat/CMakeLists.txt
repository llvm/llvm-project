add_llvm_library(LLVMBinaryFormat
  AMDGPUMetadataVerifier.cpp
  Dwarf.cpp
  Magic.cpp
  MsgPackDocument.cpp
  MsgPackDocumentYAML.cpp
  MsgPackReader.cpp
  MsgPackWriter.cpp
  Wasm.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/BinaryFormat
  )
  
