add_mlir_translation_library(MLIRSPIRVDeserialization
  DeserializeOps.cpp
  Deserializer.cpp
  Deserialization.cpp

  DEPENDS
  MLIRSPIRVSerializationGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRV
  MLIRSPIRVBinaryUtils
  MLIRSupport
  MLIRTranslation
  )


