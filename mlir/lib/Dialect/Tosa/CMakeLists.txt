add_mlir_dialect_library(MLIRTosa
  Utils/QuantUtils.cpp
  IR/TosaOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa

  DEPENDS
  MLIRStandardOpsIncGen
  MLIRTosaOpsIncGen
  MLIRTosaStructsIncGen
  MLIRTosaInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRStandard
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRQuant
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  )

add_subdirectory(Transforms)
