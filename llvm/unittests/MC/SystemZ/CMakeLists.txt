include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/SystemZ
  )

set(LLVM_LINK_COMPONENTS
  SystemZ
  MCDisassembler
  MCParser
  MC
  Support
  TargetParser
  )

add_llvm_unittest(SystemZMCTests
  SystemZAsmLexerTest.cpp
  SystemZMCDisassemblerTest.cpp
  )
