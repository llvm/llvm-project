# Main API shared library.
add_mlir_upstream_c_api_library(MLIRCAPIIR
  AffineExpr.cpp
  AffineMap.cpp
  BuiltinAttributes.cpp
  BuiltinTypes.cpp
  Diagnostics.cpp
  DialectHandle.cpp
  IntegerSet.cpp
  IR.cpp
  Pass.cpp
  Support.cpp

  LINK_LIBS PUBLIC
  MLIRBytecodeWriter
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
)
