llvm_add_library(MLIRPdllLspServerLib
  LSPServer.cpp
  PDLLServer.cpp
  MlirPdllLspServerMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-pdll-lsp-server

  LINK_LIBS PUBLIC
  MLIRPDLLParser
  MLIRLspServerSupportLib
  )
