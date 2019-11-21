
include_directories( ${LLVM_OPROFILE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMOProfileJIT
  OProfileJITEventListener.cpp
  OProfileWrapper.cpp
  )
