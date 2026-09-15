add_mlir_translation_library(MLIRSPIRVDeserialization
  DeserializeOps.cpp
  Deserializer.cpp
  Deserialization.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRVDialect
  MLIRSPIRVBinaryUtils
  MLIRSupport
  MLIRTranslateLib
  )


