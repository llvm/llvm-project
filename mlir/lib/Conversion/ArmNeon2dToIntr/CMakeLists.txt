add_mlir_conversion_library(MLIRArmNeon2dToIntr
  ArmNeon2dToIntr.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArmNeon2dToIntr

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRArmNeonDialect
  MLIRPass
  MLIRTransforms
  MLIRIR
  MLIRVectorDialect
  )
