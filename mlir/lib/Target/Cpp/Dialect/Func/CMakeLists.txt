add_mlir_translation_library(MLIRFuncToCppTranslation
  FuncToCppTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  MLIRTargetCpp
  )
