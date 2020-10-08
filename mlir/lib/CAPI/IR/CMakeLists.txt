# Main API.
add_mlir_library(MLIRCAPIIR
  AffineMap.cpp
  Diagnostics.cpp
  IR.cpp
  StandardAttributes.cpp
  StandardTypes.cpp
  Support.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
  )
