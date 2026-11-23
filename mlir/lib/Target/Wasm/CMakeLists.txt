add_mlir_translation_library(MLIRTargetWasmImport
  TranslateRegistration.cpp
  TranslateFromWasm.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/Target/Wasm

  LINK_LIBS PUBLIC
  MLIRWasmSSADialect
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
