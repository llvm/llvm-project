add_llvm_component_library(LLVMBitReader
  BitcodeAnalyzer.cpp
  BitReader.cpp
  BitcodeReader.cpp
  MetadataLoader.cpp
  ValueList.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Bitcode

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  BitstreamReader
  Core
  Support
  TargetParser
  )

# Both files define a static 'error' function that conflicts in unity builds.
set_source_files_properties(
  BitcodeReader.cpp
  MetadataLoader.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
