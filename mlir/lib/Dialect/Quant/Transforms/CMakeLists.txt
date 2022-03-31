add_mlir_dialect_library(MLIRQuantTransforms
  ConvertConst.cpp
  ConvertSimQuant.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/QuantOps/Transforms

  DEPENDS
  MLIRQuantPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRFunc
  MLIRIR
  MLIRQuant
  MLIRQuantUtils
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  )
