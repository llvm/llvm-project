add_llvm_component_library(LLVMCGData
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
  intrinsics_gen

  LINK_COMPONENTS
  BitReader
  BitWriter
  Core
  Support
  Object
  )
