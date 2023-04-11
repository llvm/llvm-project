add_mlir_translation_library(MLIRTargetKokkosCpp
  TranslateRegistration.cpp
  TranslateToKokkosCpp.cpp

  ADDITIONAL_HEADER_DIRS
  ${EMITC_MAIN_INCLUDE_DIR}/emitc/Target/KokkosCpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTranslateLib
  )
