add_subdirectory(Dialect)

  set(LLVM_OPTIONAL_SOURCES
    CppTranslationUtils.cpp
    TranslateRegistration.cpp
    TranslateToCpp.cpp
  )

  add_mlir_translation_library(MLIRTargetCpp
  TranslateToCpp.cpp
  CppTranslationUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${EMITC_MAIN_INCLUDE_DIR}/emitc/Target/Cpp

  LINK_LIBS PUBLIC
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  )

  add_mlir_translation_library(MLIRToCppTranslationRegistration
  TranslateRegistration.cpp

  LINK_LIBS PUBLIC
  MLIRBuiltinToCppTranslation
  MLIRControlFlowToCppTranslation
  MLIREmitCToCppTranslation
  MLIRFuncToCppTranslation
  MLIRMathDialect
  MLIRSCFDialect
  )
