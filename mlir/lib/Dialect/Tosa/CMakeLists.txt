add_mlir_dialect_library(MLIRTosa
  IR/TosaOps.cpp
  Utils/QuantUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa

  DEPENDS
  MLIRStandardOpsIncGen
  MLIRTosaOpsIncGen
  MLIRTosaStructsIncGen
  MLIRTosaInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  )

# add_subdirectory(Serialization)
add_subdirectory(Transforms)
