set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
)

llvm_add_library(TableGenLspServerLib
  LSPServer.cpp
  TableGenServer.cpp
  TableGenLspServerMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/tblgen-lsp-server

  LINK_LIBS PUBLIC
  MLIRLspServerSupportLib
  MLIRSupport
  )
