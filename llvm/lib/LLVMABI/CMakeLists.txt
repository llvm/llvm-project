add_llvm_component_library(LLVMABI
    ABIFunctionInfo.cpp
    Type.cpp
    Targets/x86.cpp
  
    ADDITIONAL_HEADER_DIRS
    ${LLVM_MAIN_INCLUDE_DIR}/llvm/LLVMABI
  
    LINK_COMPONENTS
    Support
  )
  