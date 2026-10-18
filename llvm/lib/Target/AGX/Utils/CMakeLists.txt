add_llvm_component_library(LLVMAGXUtils
  AGXBaseInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  AGX
  )

add_dependencies(LLVMAGXUtils AGXCommonTableGen)
