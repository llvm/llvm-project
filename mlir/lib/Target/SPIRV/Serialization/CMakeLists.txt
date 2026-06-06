add_mlir_translation_library(MLIRSPIRVSerialization
  Serialization.cpp
  Serializer.cpp
  SerializeOps.cpp

  DEPENDS
  MLIRSPIRVSerializationGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRVDialect
  MLIRSPIRVBinaryUtils
  MLIRSupport
  MLIRTranslateLib
  )
