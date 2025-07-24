add_llvm_component_library(LLVMFrontendSYCL
  OffloadWrapper.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/SYCL

  LINK_COMPONENTS
  Core
  FrontendOffloading
  Object
  Support
  TransformUtils
  )
