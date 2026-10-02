add_mlir_dialect_library(MLIRPolygeistDialect
  PolygeistOps.cpp
  PolygeistDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Polygeist

  DEPENDS
  MLIRPolygeistOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRAffineDialect
  MLIRSCFDialect
  )
