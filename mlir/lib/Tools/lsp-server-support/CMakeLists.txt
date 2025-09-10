add_mlir_library(MLIRLspServerSupportLib
  CompilationDatabase.cpp
  SourceMgrUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/lsp-server-support

  LINK_COMPONENTS
  SupportLSP

  LINK_LIBS PUBLIC
  MLIRSupport
)
