set(LLVM_TABLEGEN_PROJECT LLVM_HEADERS)
set(LLVM_TARGET_DEFINITIONS BitWriterOptions.td)
llvm_tablegen(BitWriterOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(BitWriterOptionsTableGen)

add_llvm_component_library(LLVMBitWriter
  BitWriter.cpp
  BitWriterOptions.cpp
  BitcodeWriter.cpp
  BitcodeWriterPass.cpp
  ValueEnumerator.cpp

  DEPENDS
  BitWriterOptionsTableGen
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  MC
  Object
  Option
  ProfileData
  Support
  TargetParser
  )
