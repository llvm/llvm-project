include_directories(
  ${LLVM_SOURCE_DIR}/lib/Target/VE
  ${LLVM_BINARY_DIR}/lib/Target/VE
  )

set(LLVM_LINK_COMPONENTS
  VECodeGen
  VEDesc
  VEInfo
  CodeGen
  GlobalISel
  MC
  SelectionDAG
  Support
  Target
)

add_llvm_target_unittest(VETests
  MachineInstrTest.cpp
  )
