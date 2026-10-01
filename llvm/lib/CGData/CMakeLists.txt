set(LLVM_TARGET_DEFINITIONS CGDataOptions.td)
tablegen(LLVM CGDataOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(CGDataOptionsTableGen)

add_llvm_component_library(LLVMCGData
  CGDataOptions.cpp
  CodeGenData.cpp
  CodeGenDataReader.cpp
  CodeGenDataWriter.cpp
  OutlinedHashTree.cpp
  OutlinedHashTreeRecord.cpp
  StableFunctionMap.cpp
  StableFunctionMapRecord.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CGData

  DEPENDS
  CGDataOptionsTableGen
  intrinsics_gen

  LINK_COMPONENTS
  BitReader
  BitWriter
  Core
  Support
  Object
  Option
  )
