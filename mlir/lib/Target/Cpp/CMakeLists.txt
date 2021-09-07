add_mlir_translation_library(MLIRTargetCpp
  TranslateRegistration.cpp
  TranslateToCpp.cpp

  ADDITIONAL_HEADER_DIRS
  ${EMITC_MAIN_INCLUDE_DIR}/emitc/Target/Cpp

  LINK_LIBS PUBLIC
  MLIREmitC
  MLIRIR
  MLIRSCF
  MLIRStandard
  MLIRSupport
  MLIRTranslation
  )
