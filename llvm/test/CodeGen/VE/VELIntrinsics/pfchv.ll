; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

;;; Test prefetch vector intrinsic instructions
;;;
;;; Note:
;;;   We test PFCHVrrl, PFCHVirl, PFCHVNCrrl, and PFCHVNCirl instructions.

; Function Attrs: nounwind
define void @pfchv_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: pfchv_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    pfchv %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.pfchv.ssl(i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: inaccessiblemem_or_argmemonly nounwind
declare void @llvm.ve.vl.pfchv.ssl(i64, ptr, i32)

; Function Attrs: nounwind
define void @pfchv_vssl_imm(ptr %0) {
; CHECK-LABEL: pfchv_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pfchv 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.pfchv.ssl(i64 8, ptr %0, i32 256)
  ret void
}

; Function Attrs: nounwind
define void @pfchvnc_vssl(ptr %0, i64 %1) {
; CHECK-LABEL: pfchvnc_vssl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, 256
; CHECK-NEXT:    lvl %s2
; CHECK-NEXT:    pfchv.nc %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.pfchvnc.ssl(i64 %1, ptr %0, i32 256)
  ret void
}

; Function Attrs: inaccessiblemem_or_argmemonly nounwind
declare void @llvm.ve.vl.pfchvnc.ssl(i64, ptr, i32)

; Function Attrs: nounwind
define void @pfchvnc_vssl_imm(ptr %0) {
; CHECK-LABEL: pfchvnc_vssl_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pfchv.nc 8, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  tail call void @llvm.ve.vl.pfchvnc.ssl(i64 8, ptr %0, i32 256)
  ret void
}
