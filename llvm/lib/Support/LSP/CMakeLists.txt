add_llvm_library(LLVMSupportLSP STATIC
  Protocol.cpp
  Transport.cpp
  Logging.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Support/LSP

  LINK_LIBS
  LLVMSupport
)
