add_mlir_library(MLIRLspServerLib
  LSPServer.cpp
  MLIRServer.cpp
  MlirLspServerMain.cpp
  Protocol.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-lsp-server

  LINK_COMPONENTS
  SupportLSP

  LINK_LIBS PUBLIC
  MLIRBytecodeWriter
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLspServerSupportLib
  MLIRParser
  )
