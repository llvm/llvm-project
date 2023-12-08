; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test intrinsics for communication register
;;;
;;; Note:
;;;   We test LCR, SCR, TSCR, and FIDCR instructions.

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define i64 @lcr_sss(i64 noundef %0, i64 noundef %1) {
; CHECK-LABEL: lcr_sss:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lcr %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call i64 @llvm.ve.vl.lcr.sss(i64 %0, i64 %1)
  ret i64 %3
}

; Function Attrs: nofree nosync nounwind readnone
declare i64 @llvm.ve.vl.lcr.sss(i64, i64)

; Function Attrs: nounwind
define void @scr_sss(i64 noundef %0, i64 noundef %1, i64 noundef %2) {
; CHECK-LABEL: scr_sss:
; CHECK:       # %bb.0:
; CHECK-NEXT:    scr %s0, %s1, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.scr.sss(i64 %0, i64 %1, i64 %2)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.ve.vl.scr.sss(i64, i64, i64)

; Function Attrs: nounwind
define i64 @tscr_ssss(i64 noundef %0, i64 noundef %1, i64 noundef %2) {
; CHECK-LABEL: tscr_ssss:
; CHECK:       # %bb.0:
; CHECK-NEXT:    tscr %s0, %s1, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = tail call i64 @llvm.ve.vl.tscr.ssss(i64 %0, i64 %1, i64 %2)
  ret i64 %4
}

; Function Attrs: nounwind
declare i64 @llvm.ve.vl.tscr.ssss(i64, i64, i64)

; Function Attrs: nounwind
define i64 @fidcr_ss0(i64 noundef %0) {
; CHECK-LABEL: fidcr_ss0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fidcr %s0, %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.ve.vl.fidcr.sss(i64 %0, i32 0)
  ret i64 %2
}

; Function Attrs: nounwind
declare i64 @llvm.ve.vl.fidcr.sss(i64, i32)

; Function Attrs: nounwind
define i64 @fidcr_ss1(i64 noundef %0) {
; CHECK-LABEL: fidcr_ss1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fidcr %s0, %s0, 1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.ve.vl.fidcr.sss(i64 %0, i32 1)
  ret i64 %2
}

; Function Attrs: nounwind
define i64 @fidcr_ss2(i64 noundef %0) {
; CHECK-LABEL: fidcr_ss2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fidcr %s0, %s0, 2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.ve.vl.fidcr.sss(i64 %0, i32 2)
  ret i64 %2
}

; Function Attrs: nounwind
define i64 @fidcr_ss3(i64 noundef %0) {
; CHECK-LABEL: fidcr_ss3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fidcr %s0, %s0, 3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.ve.vl.fidcr.sss(i64 %0, i32 3)
  ret i64 %2
}

; Function Attrs: nounwind
define i64 @fidcr_ss4(i64 noundef %0) {
; CHECK-LABEL: fidcr_ss4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fidcr %s0, %s0, 4
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.ve.vl.fidcr.sss(i64 %0, i32 4)
  ret i64 %2
}

; Function Attrs: nounwind
define i64 @fidcr_ss5(i64 noundef %0) {
; CHECK-LABEL: fidcr_ss5:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fidcr %s0, %s0, 5
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.ve.vl.fidcr.sss(i64 %0, i32 5)
  ret i64 %2
}

; Function Attrs: nounwind
define i64 @fidcr_ss6(i64 noundef %0) {
; CHECK-LABEL: fidcr_ss6:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fidcr %s0, %s0, 6
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.ve.vl.fidcr.sss(i64 %0, i32 6)
  ret i64 %2
}

; Function Attrs: nounwind
define i64 @fidcr_ss7(i64 noundef %0) {
; CHECK-LABEL: fidcr_ss7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fidcr %s0, %s0, 7
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.ve.vl.fidcr.sss(i64 %0, i32 7)
  ret i64 %2
}

!2 = !{!"clang version 15.0.0 (git@kaz7.github.com:sx-aurora-dev/llvm-project.git e0c5640dba6e9ba1cd29ed8d59b85c6378e48ac7)"}
