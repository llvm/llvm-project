set(LLVM_TABLEGEN_PROJECT LLVM_HEADERS)
set(LLVM_TARGET_DEFINITIONS BitReaderOptions.td)
llvm_tablegen(BitReaderOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(BitReaderOptionsTableGen)

add_llvm_component_library(LLVMBitReader
  BitcodeAnalyzer.cpp
  BitReader.cpp
  BitReaderOptions.cpp
  BitcodeReader.cpp
  MetadataLoader.cpp
  ValueList.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Bitcode

  DEPENDS
  BitReaderOptionsTableGen
  intrinsics_gen

  LINK_COMPONENTS
  BitstreamReader
  Core
  Option
  Support
  TargetParser
  )
