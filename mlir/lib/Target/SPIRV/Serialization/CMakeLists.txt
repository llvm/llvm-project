add_mlir_translation_library(MLIRSPIRVSerialization
  Serialization.cpp

  DEPENDS
  MLIRSPIRVSerializationGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRV
  MLIRSPIRVBinaryUtils
  MLIRSupport
  MLIRTranslation
  )


