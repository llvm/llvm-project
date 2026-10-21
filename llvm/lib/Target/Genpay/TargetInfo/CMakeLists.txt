include_directories( ${CMAKE_CURRENT_BINARY_DIR}/..
${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMGenpayInfo
  GenpayTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  Genpay
  )

add_dependencies(LLVMGenpayInfo GenpayCommonTableGen)
