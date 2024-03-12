add_mlir_translation_library(MLIRControlFlowToCppTranslation
  ControlFlowToCppTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowDialect
  MLIRSupport
  MLIRTargetCpp
  )
