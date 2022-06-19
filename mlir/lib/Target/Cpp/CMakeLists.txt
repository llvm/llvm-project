add_mlir_translation_library(MLIRTargetCpp
  TranslateRegistration.cpp
  TranslateToCpp.cpp

  ADDITIONAL_HEADER_DIRS
  ${EMITC_MAIN_INCLUDE_DIR}/emitc/Target/Cpp

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRControlFlowDialect
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTranslateLib
  )
