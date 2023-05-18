; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test load/save vector mask intrinsic instructions
;;;
;;; Note:
;;;   We test LVMir_m, LVMyir_y, SVMmi, and SVMyi instructions.

; Function Attrs: nounwind readnone
define i64 @lvm_mmss(ptr nocapture readnone %0, i64 %1) {
; CHECK-LABEL: lvm_mmss:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvm %vm1, 3, %s1
; CHECK-NEXT:    svm %s0, %vm1, 3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <256 x i1> @llvm.ve.vl.lvm.mmss(<256 x i1> undef, i64 3, i64 %1)
  %4 = tail call i64 @llvm.ve.vl.svm.sms(<256 x i1> %3, i64 3)
  ret i64 %4
}

; Function Attrs: nounwind readnone
declare <256 x i1> @llvm.ve.vl.lvm.mmss(<256 x i1>, i64, i64)

; Function Attrs: nounwind readnone
declare i64 @llvm.ve.vl.svm.sms(<256 x i1>, i64)

; Function Attrs: nounwind readnone
define i64 @lvml_MMss(ptr nocapture readnone %0, i64 %1) {
; CHECK-LABEL: lvml_MMss:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lvm %vm2, 1, %s1
; CHECK-NEXT:    svm %s0, %vm3, 3
; CHECK-NEXT:    svm %s1, %vm2, 2
; CHECK-NEXT:    adds.l %s0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call <512 x i1> @llvm.ve.vl.lvm.MMss(<512 x i1> undef, i64 5, i64 %1)
  %4 = tail call i64 @llvm.ve.vl.svm.sMs(<512 x i1> %3, i64 3)
  %5 = tail call i64 @llvm.ve.vl.svm.sMs(<512 x i1> %3, i64 6)
  %6 = add i64 %5, %4
  ret i64 %6
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.ve.vl.lvm.MMss(<512 x i1>, i64, i64)

; Function Attrs: nounwind readnone
declare i64 @llvm.ve.vl.svm.sMs(<512 x i1>, i64)
