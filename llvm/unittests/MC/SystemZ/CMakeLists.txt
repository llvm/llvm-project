include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/SystemZ
  )

set(LLVM_LINK_COMPONENTS
  SystemZ
  MCParser
  MC
  Support
  TargetParser
  )

add_llvm_unittest(SystemZAsmLexerTests
  SystemZAsmLexerTest.cpp
  )
