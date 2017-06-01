add_llvm_library(LLVMXRay
  Trace.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ADT
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/XRay

  DEPENDS
  LLVMSupport

  LINK_LIBS
  LLVMSupport
  )
