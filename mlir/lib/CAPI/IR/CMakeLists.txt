# Main API shared library.
add_mlir_public_c_api_library(MLIRCAPIIR
  AffineExpr.cpp
  AffineMap.cpp
  Diagnostics.cpp
  IR.cpp
  Pass.cpp
  StandardAttributes.cpp
  StandardTypes.cpp
  Support.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
)
