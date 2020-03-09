add_mlir_dialect_library(MLIRSPIRVSerialization
  Deserializer.cpp
  Serializer.cpp
  SPIRVBinaryUtils.cpp
  TranslateRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV

  DEPENDS
  MLIRSPIRVSerializationGen
  )
target_link_libraries(MLIRSPIRVSerialization
  PUBLIC
  MLIRIR
  MLIRSPIRV
  MLIRSupport
  MLIRTranslation
  )
