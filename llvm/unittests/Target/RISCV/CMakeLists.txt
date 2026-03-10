include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV
  ${LLVM_BINARY_DIR}/lib/Target/RISCV
  )

set(LLVM_LINK_COMPONENTS
  RISCVCodeGen
  RISCVDesc
  RISCVInfo
  Analysis
  AsmParser
  CodeGen
  Core
  MC
  Passes
  SelectionDAG
  TargetParser
  )

add_llvm_target_unittest(RISCVTests
  MCInstrAnalysisTest.cpp
  RISCVInstrInfoTest.cpp
  RISCVSelectionDAGTest.cpp
  )
