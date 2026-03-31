add_llvm_component_library(LLVMABI
  Types.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ABI

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
)

