include_directories( ${CMAKE_CURRENT_BINARY_DIR}/..
                     ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMP2Info
  P2TargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  P2
)

add_dependencies(LLVMP2Info P2CommonTableGen)
