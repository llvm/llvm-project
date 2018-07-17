add_llvm_library(LLVMDemangle
  ItaniumDemangle.cpp
  
  ADDITIONAL_HEADER_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}/llvm/Demangle"
)
