add_llvm_component_library(LLVMFrontendSYCL
  SplitModule.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/SYCL

  LINK_COMPONENTS
  Core
  Support
  TransformUtils
  )
