; RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=asm %s | FileCheck %s

; FIXME: Currently we can't print register names in CFI directives
; without extending MC to support DWARF register names that are distinct
; from physical register names.

.text
f:
.cfi_startproc
; CHECK: .cfi_undefined 2560
.cfi_undefined 2560
; FIXME: Until we implement a distinct set of DWARF register names we
; will continue to parse physical registers and pick an arbitrary encoding.
; CHECK: .cfi_undefined 2560
.cfi_undefined v0
.cfi_endproc
