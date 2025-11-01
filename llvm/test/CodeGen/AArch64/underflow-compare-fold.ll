; RUN: llc < %s -mtriple=aarch64 | FileCheck %s

; GitHub issue #161036

; Positive test : umin(sub(a,b),a) with scalar types should be folded
define i64 @underflow_compare_fold(i64 %a, i64 %b) {
; CHECK-LABEL: underflow_compare_fold
; CHECK:      // %bb.0:
; CHECK-NEXT: subs x8, x0, x1
; CHECK-NEXT: csel x0, x0, x8, lo
; CHECK-NEXT: ret
  %sub = sub i64 %a, %b
  %cond = tail call i64 @llvm.umin.i64(i64 %sub, i64 %a)
  ret i64 %cond
}

; Negative test, vector types : umin(sub(a,b),a) but with vectors
define <16 x i8> @underflow_compare_dontfold_vectors(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: underflow_compare_dontfold_vectors
; CHECK-LABEL: %bb.0
; CHECK-NEXT: sub v1.16b, v0.16b, v1.16b
; CHECK-NEXT: umin v0.16b, v1.16b, v0.16b
; CHECK-NEXT: ret
  %sub = sub <16 x i8> %a, %b
  %cond = tail call <16 x i8> @llvm.umin.v16i8(<16 x i8> %sub, <16 x i8> %a)
  ret <16 x i8> %cond
}

; Negative test, pattern mismatch : umin(a,sub(a,b))
define i64 @umin_sub_inverse_args(i64 %a, i64 %b) {
; CHECK-LABEL: umin_sub_inverse_args
; CHECK-LABEL: %bb.0
; CHECK-NEXT: sub x8, x0, x1
; CHECK-NEXT: cmp x0, x8
; CHECK-NEXT: csel x0, x0, x8, lo
; CHECK-NEXT: ret
  %sub = sub i64 %a, %b
  %cond = tail call i64 @llvm.umin.i64(i64 %a, i64 %sub)
  ret i64 %cond
}

; Negative test, pattern mismatch : umin(add(a,b),a)
define i64 @umin_add(i64 %a, i64 %b) {
; CHECK-LABEL: umin_add
; CHECK-LABEL: %bb.0
; CHECK-NEXT: add x8, x0, x1
; CHECK-NEXT: cmp x8, x0
; CHECK-NEXT: csel x0, x8, x0, lo
; CHECK-NEXT: ret
  %add = add i64 %a, %b
  %cond = tail call i64 @llvm.umin.i64(i64 %add, i64 %a)
  ret i64 %cond
}
