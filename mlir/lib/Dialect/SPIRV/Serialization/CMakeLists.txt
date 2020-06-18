add_mlir_dialect_library(MLIRSPIRVSerialization
  Deserializer.cpp
  Serializer.cpp
  SPIRVBinaryUtils.cpp
  TranslateRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV

  DEPENDS
  MLIRSPIRVSerializationGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRV
  MLIRSupport
  MLIRTranslation
  )
