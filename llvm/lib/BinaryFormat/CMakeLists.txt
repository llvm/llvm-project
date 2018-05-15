add_llvm_library(LLVMBinaryFormat
  Dwarf.cpp
  Magic.cpp
  Wasm.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/BinaryFormat
  )
  
