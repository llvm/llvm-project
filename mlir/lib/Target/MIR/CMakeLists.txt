add_mlir_translation_library(MLIRTargetMIRImport
  MIRImport.cpp
  MIRExport.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/MIR

  LINK_COMPONENTS
  AArch64CodeGen
  AArch64Info
  AArch64Desc
  CodeGen
  Core
  MIRParser
  Target
  TargetParser

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAArch64MIRDialect
  MLIRMIRDialect
  MLIRTranslateLib
  )
