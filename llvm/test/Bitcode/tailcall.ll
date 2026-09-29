; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s

; Check that musttail and tail roundtrip.

declare cc1023 void @t1_callee()
define cc1023 void @t1() {
; CHECK: tail call cc1023 void @t1_callee()
  tail call cc1023 void @t1_callee()
  ret void
}

declare cc1023 void @t2_callee()
define cc1023 void @t2() {
; CHECK: musttail call cc1023 void @t2_callee()
  musttail call cc1023 void @t2_callee()
  ret void
}
