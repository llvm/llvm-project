add_llvm_component_library(LLVMDWARFLinkerBase
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DWARFLinker

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  )

add_subdirectory(Apple)
add_subdirectory(LLVM)
