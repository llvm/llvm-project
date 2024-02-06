add_mlir_conversion_library(MLIRTosaToMLProgram
  TosaToMLProgram.cpp
  TosaToMLProgramPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMLProgramDialect
  MLIRPass
  MLIRTosaDialect
  MLIRTosaTransforms
  MLIRSupport
  )
