add_subdirectory(Deserialization)
add_subdirectory(Serialization)

set(LLVM_OPTIONAL_SOURCES
  SPIRVBinaryUtils.cpp
  TranslateRegistration.cpp
  Target.cpp
  )

add_mlir_translation_library(MLIRSPIRVBinaryUtils
  SPIRVBinaryUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  )

add_mlir_translation_library(MLIRSPIRVTranslateRegistration
  TranslateRegistration.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRVSerialization
  MLIRSPIRVDeserialization
  MLIRSupport
  )

add_mlir_dialect_library(MLIRSPIRVTarget
  Target.cpp

  # Target.cpp includes llvm/IR/Attributes.h.
  DEPENDS
  intrinsics_gen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRGPUDialect
  MLIRSPIRVSerialization
  MLIRSupport
  )
