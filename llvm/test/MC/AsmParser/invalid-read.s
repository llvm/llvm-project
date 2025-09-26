# RUN: llvm-mc -triple=x86_64 --as-lex %s

# This test ensures AsmLexer doesn't perform an invalid read in a case where
# buffer ends with '\0', ' ' or '\t'
  