add_llvm_component_library(LLVMOption
  Arg.cpp
  ArgList.cpp
  Option.cpp
  OptionTable.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Option

  LINK_COMPONENTS
  Support
  )
