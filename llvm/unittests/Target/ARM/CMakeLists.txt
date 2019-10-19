include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/ARM
  ${LLVM_BINARY_DIR}/lib/Target/ARM
  )

set(LLVM_LINK_COMPONENTS
  ARMCodeGen
  ARMDesc
  ARMInfo
  CodeGen
  GlobalISel
  MC
  SelectionDAG
  Support
  Target
)

add_llvm_unittest(ARMTests
  MachineInstrTest.cpp
  )
