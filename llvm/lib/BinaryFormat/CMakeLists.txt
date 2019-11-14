add_llvm_component_library(LLVMBinaryFormat
  AMDGPUMetadataVerifier.cpp
  Dwarf.cpp
  Magic.cpp
  Minidump.cpp
  MsgPackDocument.cpp
  MsgPackDocumentYAML.cpp
  MsgPackReader.cpp
  MsgPackWriter.cpp
  Wasm.cpp
  XCOFF.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/BinaryFormat
  )
  
