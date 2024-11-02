add_mlir_library(MLIRLspServerSupportLib
  CompilationDatabase.cpp
  Logging.cpp
  Protocol.cpp
  SourceMgrUtils.cpp
  Transport.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/lsp-server

  LINK_LIBS PUBLIC
  MLIRSupport
  )
