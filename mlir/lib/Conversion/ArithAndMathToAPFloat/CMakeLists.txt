add_mlir_library(MLIRArithAndMathToAPFloatUtils
  Utils.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRVectorDialect
  )

add_mlir_conversion_library(MLIRArithToAPFloat
  ArithToAPFloat.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithAndMathToAPFloatUtils
  MLIRArithDialect
  MLIRArithTransforms
  MLIRFuncDialect
  MLIRFuncUtils
  MLIRVectorDialect
  )

add_mlir_conversion_library(MLIRMathToAPFloat
  MathToAPFloat.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MathToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  MLIRArithAndMathToAPFloatUtils
  MLIRMathDialect
  MLIRFuncDialect
  MLIRFuncUtils
  MLIRVectorDialect
  )
