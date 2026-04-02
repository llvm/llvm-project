add_mlir_library(MLIRPDLLCodeGen
  CPPGen.cpp
  MLIRGen.cpp

  LINK_LIBS PUBLIC
  MLIRParser
  MLIRPDLLAST
  MLIRPDLDialect
  MLIRSupport
  )

# Static PDLL code generation helpers and symbol registration conflicts.
set_source_files_properties(
  MLIRGen.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
