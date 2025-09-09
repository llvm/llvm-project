set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  SupportLSP
  TableGen
)

llvm_add_library(TableGenLspServerLib
  LSPServer.cpp
  TableGenServer.cpp
  TableGenLspServerMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/tblgen-lsp-server

  DISABLE_LLVM_LINK_LLVM_DYLIB

  LINK_LIBS PUBLIC
  MLIRLspServerSupportLib
  MLIRSupport
  )
