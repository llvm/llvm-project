add_mlir_dialect_library(MLIRStandardOps
  IR/Ops.cpp
  EDSC/Builders.cpp
  EDSC/Intrinsics.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/StandardOps

  DEPENDS
  MLIRCallOpInterfacesIncGen
  MLIRStandardOpsIncGen

  LINK_LIBS
  MLIREDSC
  MLIRIR
  MLIRSupport
  LLVMCore
  LLVMSupport
  )
