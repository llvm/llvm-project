add_mlir_upstream_c_api_library(MLIRCAPITarget
  LLVMIR.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
)

add_mlir_upstream_c_api_library(MLIRCAPIExportSMTLIB
  ExportSMTLIB.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRExportSMTLIB
)
