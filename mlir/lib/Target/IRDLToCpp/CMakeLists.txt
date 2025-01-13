add_mlir_translation_library(MLIRTargetIRDLToCpp
  TranslationRegistration.cpp
  IRDLToCpp.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIRDL
  MLIRTranslateLib
  )
