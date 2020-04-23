; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=future -ppc-asm-full-reg-names < %s | FileCheck %s

; The test checks the behavior of PC Relative indirect calls. When using
; PC Relative, TOC save and restore are no longer required. Function pointer
; is passed as a parameter in this test.

; Function Attrs: noinline
define dso_local void @IndirectCallExternFuncPtr(void ()* nocapture %ptrfunc) {
; CHECK-LABEL: IndirectCallExternFuncPtr:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mflr r0
; CHECK-NEXT:    std r0, 16(r1)
; CHECK-NEXT:    stdu r1, -32(r1)

; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    .cfi_offset lr, 16
; CHECK-NEXT:    mtctr r3
; CHECK-NEXT:    mr r12, r3
; CHECK-NEXT:    bctrl

; CHECK-NEXT:    addi r1, r1, 32
; CHECK-NEXT:    ld r0, 16(r1)
; CHECK-NEXT:    mtlr r0
; CHECK-NEXT:    blr
entry:
  tail call void %ptrfunc()
  ret void
}

define dso_local void @FuncPtrPassAsParam() {
entry:
  tail call void @IndirectCallExternFuncPtr(void ()* nonnull @Function)
  ret void
}

declare void @Function()
