add_mlir_library(MLIRAllocationOpInterface
  AllocationOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization/IR

  DEPENDS
  MLIRAllocationOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
