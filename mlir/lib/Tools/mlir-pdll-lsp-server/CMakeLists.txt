llvm_add_library(MLIRPdllLspServerLib
  LSPServer.cpp
  PDLLServer.cpp
  Protocol.cpp
  MlirPdllLspServerMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-pdll-lsp-server

  # MLIRPDLLParser needs LLVMTableGen, which is excluded from libLLVM.so.
  DISABLE_LLVM_LINK_LLVM_DYLIB

  LINK_COMPONENTS
  SupportLSP

  LINK_LIBS PUBLIC
  MLIRPDLLCodeGen
  MLIRPDLLParser
  MLIRLspServerSupportLib
  )
