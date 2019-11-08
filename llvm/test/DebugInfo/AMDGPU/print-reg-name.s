; RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=asm %s | FileCheck %s

; Check that we can print symbolic register operands in CFI instructions.

.text
f:
.cfi_startproc
; CHECK: .cfi_undefined s0
.cfi_undefined s0
.cfi_endproc
