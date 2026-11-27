add_llvm_component_library(LLVMDWP
  DWP.cpp
  DWPError.cpp
  ELFWriter.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DWP

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  DebugInfoDWARF
  Object
  Support
  )
