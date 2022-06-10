add_mlir_dialect_library(MLIRTosa
  Utils/ConversionUtils.cpp
  Utils/QuantUtils.cpp
  IR/TosaOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa

  DEPENDS
  MLIRTosaAttributesIncGen
  MLIRTosaOpsIncGen
  MLIRTosaInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRQuant
  MLIRQuantUtils
  MLIRSideEffectInterfaces
  MLIRTensor
  MLIRViewLikeInterface
  )

add_subdirectory(Transforms)
