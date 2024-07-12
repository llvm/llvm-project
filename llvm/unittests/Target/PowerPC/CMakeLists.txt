include_directories(
  ${PROJECT_SOURCE_DIR}/lib/Target/PowerPC
  ${PROJECT_BINARY_DIR}/lib/Target/PowerPC
  )

set(LLVM_LINK_COMPONENTS
  MC
  Support
  Target
  TargetParser
  PowerPCCodeGen
  PowerPCDesc
  PowerPCInfo
  )

add_llvm_unittest(PowerPCTests
  AIXRelocModelTest.cpp
  )
