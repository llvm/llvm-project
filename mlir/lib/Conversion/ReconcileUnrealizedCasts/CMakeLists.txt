add_mlir_conversion_library(MLIRReconcileUnrealizedCasts
  ReconcileUnrealizedCasts.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ReconcileUnrealizedCasts

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
  )
