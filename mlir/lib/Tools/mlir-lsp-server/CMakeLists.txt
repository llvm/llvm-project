add_mlir_library(MLIRLspServerLib
  lsp/Logging.cpp
  lsp/Protocol.cpp
  lsp/Transport.cpp
  LSPServer.cpp
  MLIRServer.cpp
  MlirLspServerMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-lsp-server

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  )
