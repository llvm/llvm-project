add_llvm_component_library(LLVMABI
  QualTypeMapper.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ABI

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
)

target_include_directories(LLVMABI PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
)
