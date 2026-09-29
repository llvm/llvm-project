; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test leading zero of vm intrinsic instructions
;;;
;;; Note:
;;;   We test LZVM*ml instruction.

; Function Attrs: nounwind readnone
define fastcc i64 @lzvm_sml(<256 x i1> %0) {
; CHECK-LABEL: lzvm_sml:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    lzvm %s0, %vm1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.ve.vl.lzvm.sml(<256 x i1> %0, i32 256)
  ret i64 %2
}

; Function Attrs: nounwind readnone
declare i64 @llvm.ve.vl.lzvm.sml(<256 x i1>, i32)
