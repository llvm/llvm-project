include_directories(
  ${CMAKE_SOURCE_DIR}/lib/Target/PowerPC
  ${CMAKE_BINARY_DIR}/lib/Target/PowerPC
  )

set(LLVM_LINK_COMPONENTS
  MC
  Support
  Target
  PowerPCCodeGen
  PowerPCDesc
  PowerPCInfo
  )

add_llvm_unittest(PowerPCTests
  AIXRelocModelTest.cpp
  )

set_property(TARGET PowerPCTests PROPERTY FOLDER "Tests/UnitTests/TargetTests")
