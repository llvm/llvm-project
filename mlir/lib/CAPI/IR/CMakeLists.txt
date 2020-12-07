# Main API shared library.
add_mlir_public_c_api_library(MLIRCAPIIR
  AffineExpr.cpp
  AffineMap.cpp
  BuiltinAttributes.cpp
  BuiltinTypes.cpp
  Diagnostics.cpp
  IR.cpp
  Pass.cpp
  Support.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
)
