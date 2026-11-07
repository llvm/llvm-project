add_mlir_library(MLIRFuzzMutate
  MLIRMutator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/FuzzMutate

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRUBDialect
  )
