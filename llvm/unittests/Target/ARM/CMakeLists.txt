include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/ARM
  ${LLVM_BINARY_DIR}/lib/Target/ARM
  )

set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  ARMCodeGen
  ARMDesc
  ARMInfo
  AsmParser
  CodeGen
  GlobalISel
  MC
  SelectionDAG
  Support
  Target
)

add_llvm_target_unittest(ARMTests
  ARMSelectionDAGTest.cpp
  MachineInstrTest.cpp
  )
