add_subdirectory(Deserialization)
add_subdirectory(Serialization)

set(LLVM_OPTIONAL_SOURCES
  SPIRVBinaryUtils.cpp
  TranslateRegistration.cpp
  )

add_mlir_translation_library(MLIRSPIRVBinaryUtils
  SPIRVBinaryUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRV
  MLIRSupport
  )

add_mlir_translation_library(MLIRSPIRVTranslateRegistration
  TranslateRegistration.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRSPIRVDeserialization
  MLIRSupport
  MLIRTranslateLib
  )
