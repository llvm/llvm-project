add_mlir_translation_library(MLIRExportSMTLIB
  ExportSMTLIB.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRSMT
  MLIRSupport
  MLIRFuncDialect
  MLIRIR
  MLIRTranslateLib
)
