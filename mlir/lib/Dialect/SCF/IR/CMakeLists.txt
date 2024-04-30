add_mlir_dialect_library(MLIRSCFDialect
  SCF.cpp
  DeviceMappingInterface.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/SCF

  DEPENDS
  MLIRSCFOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRDialectUtils
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRValueBoundsOpInterface
  )

