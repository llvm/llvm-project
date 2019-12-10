add_llvm_component_library(LLVMFrontendOpenMP
  OMPConstants.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/OpenMP

  DEPENDS
  intrinsics_gen
  )
