; RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1010 -filetype=obj %s | llvm-dwarfdump -debug-frame - | FileCheck %s

; Check that we implement the DWARF register mapping.

.text
f:
.cfi_startproc
; CHECK: CIE
; CHECK: Return address column: 16

; CHECK: FDE
; CHECK: DW_CFA_undefined: reg16
.cfi_undefined pc
; CHECK: DW_CFA_undefined: reg17
.cfi_undefined exec

; CHECK: DW_CFA_undefined: reg32
.cfi_undefined s0
; CHECK: DW_CFA_undefined: reg95
.cfi_undefined s63

; CHECK: DW_CFA_undefined: reg1088
.cfi_undefined s64
; CHECK: DW_CFA_undefined: reg1129
.cfi_undefined s105

; CHECK: DW_CFA_undefined: reg2560
.cfi_undefined v0
; CHECK: DW_CFA_undefined: reg2815
.cfi_undefined v255

; CHECK: DW_CFA_undefined: reg3072
.cfi_undefined a0
; CHECK: DW_CFA_undefined: reg3327
.cfi_undefined a255

.cfi_endproc
