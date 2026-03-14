; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -target-abi=elfv2 -mtriple=powerpc64-- \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | FileCheck %s


; The test checks the behavior of PC Relative indirect calls. When using
; PC Relative, TOC save and restore are no longer required. Function pointer
; is passed as a parameter in this test.

; Function Attrs: noinline
define dso_local void @IndirectCallExternFuncPtr(ptr nocapture %ptrfunc) {
; CHECK-LABEL: IndirectCallExternFuncPtr:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mtctr r3
; CHECK-NEXT:    mr r12, r3
; CHECK-NEXT:    bctr
; CHECK-NEXT:    #TC_RETURNr8 ctr
entry:
  tail call void %ptrfunc()
  ret void
}

define dso_local void @FuncPtrPassAsParam() {
entry:
  tail call void @IndirectCallExternFuncPtr(ptr nonnull @Function)
  ret void
}

declare void @Function()
