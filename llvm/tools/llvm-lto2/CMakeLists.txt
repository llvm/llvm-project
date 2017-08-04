set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BitReader
  Core
  Linker
  LTO
  MC
  Object
  Support
  Target
  )

add_llvm_tool(llvm-lto2
  llvm-lto2.cpp

  DEPENDS
  intrinsics_gen
  )
