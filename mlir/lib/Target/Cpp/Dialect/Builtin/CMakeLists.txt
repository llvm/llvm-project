add_mlir_translation_library(MLIRBuiltinToCppTranslation
  BuiltinToCppTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTargetCpp
  )
