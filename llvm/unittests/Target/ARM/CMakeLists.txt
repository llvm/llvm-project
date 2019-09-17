include_directories(
  ${CMAKE_SOURCE_DIR}/lib/Target/ARM
  ${CMAKE_BINARY_DIR}/lib/Target/ARM
  )

set(LLVM_LINK_COMPONENTS
  ARMCodeGen
  ARMDesc
  ARMInfo
  MC
  Support
  Target
)

add_llvm_unittest(ARMTests
  MachineInstrTest.cpp
  )
