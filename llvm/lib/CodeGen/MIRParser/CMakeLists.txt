add_llvm_component_library(LLVMMIRParser
  MILexer.cpp
  MIParser.cpp
  MIRParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CodeGen/MIRParser

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  BinaryFormat
  CodeGen
  CodeGenTypes
  Core
  MC
  Support
  Target
  )

# Static functions and anonymous namespace classes conflict in unity builds.
set_source_files_properties(
  MIParser.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
