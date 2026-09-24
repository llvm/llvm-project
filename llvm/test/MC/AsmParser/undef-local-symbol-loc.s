# RUN: not llvm-mc -triple=x86_64-apple-macos %s 2>&1 | FileCheck %s

.text
.globl _foo
_foo:
# CHECK: :[[#@LINE+2]]:9: error: assembler local symbol 'Ltmp_undefined' not defined
# CHECK-NEXT:   jmp Ltmp_undefined
    jmp Ltmp_undefined
    nop
    jmp Ltmp_undefined
