include_directories(
  ${CMAKE_SOURCE_DIR}/lib/Target/AArch64
  ${CMAKE_BINARY_DIR}/lib/Target/AArch64
  )

set(LLVM_LINK_COMPONENTS
  AArch64CodeGen
  AArch64Desc
  AArch64Info
  CodeGen
  Core
  GlobalISel
  MC
  MIRParser
  SelectionDAG
  Support
  Target
  )

add_llvm_unittest(AArch64Tests
  InstSizes.cpp
  )
