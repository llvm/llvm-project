add_mlir_conversion_library(MLIREmitCCommonConversion
  TypeConverter.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/EmitCCommon

  LINK_LIBS PUBLIC
  MLIREmitCDialect
  MLIRIR
  MLIRTransforms
  )
