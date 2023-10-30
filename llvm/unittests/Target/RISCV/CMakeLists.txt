include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV
  ${LLVM_BINARY_DIR}/lib/Target/RISCV
  )

set(LLVM_LINK_COMPONENTS
  RISCVCodeGen
  RISCVDesc
  RISCVInfo
  TargetParser
  MC
  )

add_llvm_target_unittest(RISCVTests
  MCInstrAnalysisTest.cpp
  RISCVBaseInfoTest.cpp
  )

set_property(TARGET RISCVTests PROPERTY FOLDER "Tests/UnitTests/TargetTests")
