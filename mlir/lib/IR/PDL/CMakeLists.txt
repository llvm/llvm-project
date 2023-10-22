add_mlir_library(MLIRIRPDLPatternMatch
  PDLPatternMatch.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
)

