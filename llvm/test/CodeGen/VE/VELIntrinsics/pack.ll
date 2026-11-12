; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test pack intrinsic instructions
;;;
;;; Note:
;;;   We test pack_f32p and pack_f32a pseudo instruction.

; Function Attrs: nounwind readonly
define fastcc i64 @pack_f32p(ptr readonly %0, ptr readonly %1) {
; CHECK-LABEL: pack_f32p:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldu %s0, (, %s0)
; CHECK-NEXT:    ldl.zx %s1, (, %s1)
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call i64 @llvm.ve.vl.pack.f32p(ptr %0, ptr %1)
  ret i64 %3
}

; Function Attrs: nounwind readonly
declare i64 @llvm.ve.vl.pack.f32p(ptr, ptr)

; Function Attrs: nounwind readonly
define fastcc i64 @pack_f32a(ptr readonly %0) {
; CHECK-LABEL: pack_f32a:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.zx %s0, (, %s0)
; CHECK-NEXT:    lea %s1, 1
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, 1(, %s1)
; CHECK-NEXT:    mulu.l %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.ve.vl.pack.f32a(ptr %0)
  ret i64 %2
}

; Function Attrs: nounwind readonly
declare i64 @llvm.ve.vl.pack.f32a(ptr)
