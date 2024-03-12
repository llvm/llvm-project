add_mlir_translation_library(MLIREmitCToCppTranslation
  EmitCToCppTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIREmitCDialect
  MLIRSupport
  MLIRTargetCpp
  )
