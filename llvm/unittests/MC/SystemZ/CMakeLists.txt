include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/SystemZ
  )

set(LLVM_LINK_COMPONENTS
  SystemZ
  MC
  Support
  )

add_llvm_unittest(SystemZAsmLexer
  SystemZAsmLexerTest.cpp
  )
