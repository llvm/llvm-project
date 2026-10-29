include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_BINARY_DIR}/lib/Target/X86
  )

set(LLVM_LINK_COMPONENTS
  MC
  MCDisassembler
  TargetParser
  X86Desc
  X86Disassembler
  X86Info
  )

add_llvm_unittest(X86MCTests
  X86MCInstrAnalysisTest.cpp
  X86MCDisassemblerTest.cpp
  )
