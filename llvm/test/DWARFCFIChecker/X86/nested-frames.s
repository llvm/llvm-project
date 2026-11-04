# RUN: not llvm-mc -triple x86_64-unknown-unknown %s --validate-cfi --filetype=null 2>&1 | FileCheck %s
# RUN: llvm-mc -triple x86_64-unknown-unknown %s --filetype=asm 2>&1 | FileCheck %s -check-prefix=ASSEMBLER --implicit-check-not warning: --implicit-check-not error:
# ASSEMBLER: .section
.pushsection A
f: 
.cfi_startproc
.cfi_same_value %rbp
.cfi_same_value %rsi
addq $10, %rbp
# CHECK: error: changed register RBP, that register RBP's unwinding rule uses, but there is no CFI directives about it
nop
.cfi_undefined %rbp

.pushsection B
g: 
.cfi_startproc
.cfi_same_value %rbp
addq $10, %rbp
# CHECK: error: changed register RBP, that register RBP's unwinding rule uses, but there is no CFI directives about it
nop
.cfi_undefined %rsi
ret
.cfi_endproc
.popsection

addq $10, %rsi
# CHECK: error: changed register RSI, that register RSI's unwinding rule uses, but there is no CFI directives about it
ret
.cfi_endproc
.popsection
