; RUN: llc < %s -mtriple=aarch64 | FileCheck %s

; GitHub issue #161036

; Positive test : umin(sub(a,b),a) with scalar types should be folded
define i64 @underflow_compare_fold_i64(i64 %a, i64 %b) {
; CHECK-LABEL: underflow_compare_fold_i64
; CHECK-LABEL: %bb.0:
; CHECK-NEXT:  subs x8, x0, x1
; CHECK-NEXT:  csel x0, x0, x8, lo
; CHECK-NEXT:  ret
  %sub = sub i64 %a, %b
  %cond = tail call i64 @llvm.umin.i64(i64 %sub, i64 %a)
  ret i64 %cond
}

; Positive test : umin(a,sub(a,b)) with scalar types should be folded
define i64 @underflow_compare_fold_i64_commute(i64 %a, i64 %b) {
; CHECK-LABEL: underflow_compare_fold_i64_commute
; CHECK-LABEL: %bb.0:
; CHECK-NEXT:  subs x8, x0, x1
; CHECK-NEXT:  csel x0, x0, x8, lo
; CHECK-NEXT:  ret
  %sub = sub i64 %a, %b
  %cond = tail call i64 @llvm.umin.i64(i64 %a, i64 %sub)
  ret i64 %cond
}

; Positive test : multi-use is OK since the sub instruction still runs once
define i64 @underflow_compare_fold_i64_multi_use(i64 %a, i64 %b, ptr addrspace(1) %ptr) {
; CHECK-LABEL: underflow_compare_fold_i64_multi_use
; CHECK-LABEL: %bb.0:
; CHECK-NEXT:  subs x8, x0, x1
; CHECK-NEXT:  csel x0, x0, x8, lo
; CHECK-NEXT:  str	x8, [x2]
; CHECK-NEXT:  ret
  %sub = sub i64 %a, %b
  store i64 %sub, ptr addrspace(1) %ptr
  %cond = call i64 @llvm.umin.i64(i64 %sub, i64 %a)
  ret i64 %cond
}

; Positive test : i32
define i32 @underflow_compare_fold_i32(i32 %a, i32 %b) {
; CHECK-LABEL: underflow_compare_fold_i32
; CHECK-LABEL: %bb.0:
; CHECK-NEXT:  subs w8, w0, w1
; CHECK-NEXT:  csel w0, w0, w8, lo
; CHECK-NEXT:  ret
  %sub = sub i32 %a, %b
  %cond = tail call i32 @llvm.umin.i32(i32 %sub, i32 %a)
  ret i32 %cond
}

; Positive test : i32
define i32 @underflow_compare_fold_i32_commute(i32 %a, i32 %b) {
; CHECK-LABEL: underflow_compare_fold_i32_commute
; CHECK-LABEL: %bb.0:
; CHECK-NEXT:  subs w8, w0, w1
; CHECK-NEXT:  csel w0, w0, w8, lo
; CHECK-NEXT:  ret
  %sub = sub i32 %a, %b
  %cond = tail call i32 @llvm.umin.i32(i32 %a, i32 %sub)
  ret i32 %cond
}

; Positive test : i32
define i32 @underflow_compare_fold_i32_multi_use(i32 %a, i32 %b, ptr addrspace(1) %ptr) {
; CHECK-LABEL: underflow_compare_fold_i32_multi_use
; CHECK-LABEL: %bb.0:
; CHECK-NEXT:  subs w8, w0, w1
; CHECK-NEXT:  csel w0, w0, w8, lo
; CHECK-NEXT:  str	w8, [x2]
; CHECK-NEXT:  ret
  %sub = sub i32 %a, %b
  store i32 %sub, ptr addrspace(1) %ptr
  %cond = call i32 @llvm.umin.i32(i32 %sub, i32 %a)
  ret i32 %cond
}

; Negative test : i16
define i16 @underflow_compare_fold_i16(i16 %a, i16 %b) {
; CHECK-LABEL: underflow_compare_fold_i16
; CHECK-LABEL: %bb.0:
; CHECK-LABEL: sub w8, w0, w1
; CHECK-LABEL: and w9, w0, #0xffff
; CHECK-LABEL: and w8, w8, #0xffff
; CHECK-LABEL: cmp w8, w9
; CHECK-LABEL: csel w0, w8, w9, lo
; CHECK-LABEL: ret
  %sub = sub i16 %a, %b
  %cond = tail call i16 @llvm.umin.i16(i16 %sub, i16 %a)
  ret i16 %cond
}

; Negative test : i16
define i16 @underflow_compare_fold_i16_commute(i16 %a, i16 %b) {
; CHECK-LABEL: underflow_compare_fold_i16_commute
; CHECK-LABEL: %bb.0:
; CHECK-LABEL: sub w8, w0, w1
; CHECK-LABEL: and w9, w0, #0xffff
; CHECK-LABEL: and w8, w8, #0xffff
; CHECK-LABEL: cmp w9, w8
; CHECK-LABEL: csel w0, w9, w8, lo
; CHECK-LABEL: ret
  %sub = sub i16 %a, %b
  %cond = tail call i16 @llvm.umin.i16(i16 %a, i16 %sub)
  ret i16 %cond
}

; Negative test : i16
define i16 @underflow_compare_fold_i16_multi_use(i16 %a, i16 %b, ptr addrspace(1) %ptr) {
; CHECK-LABEL: underflow_compare_fold_i16_multi_use
; CHECK-LABEL: %bb.0:
; CHECK-LABEL: sub w8, w0, w1
; CHECK-LABEL: and w9, w0, #0xffff
; CHECK-LABEL: and w10, w8, #0xffff
; CHECK-LABEL: strh w8, [x2]
; CHECK-LABEL: cmp w10, w9
; CHECK-LABEL: csel w0, w10, w9, lo
; CHECK-LABEL: ret
  %sub = sub i16 %a, %b
  store i16 %sub, ptr addrspace(1) %ptr
  %cond = call i16 @llvm.umin.i16(i16 %sub, i16 %a)
  ret i16 %cond
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
