# RUN: printf 'ret\0 ' > %t.s
# RUN: llvm-mc %t.s --triple=x86_64 --as-lex | FileCheck %s

# CHECK-NOT: ERROR: AddressSanitizer
