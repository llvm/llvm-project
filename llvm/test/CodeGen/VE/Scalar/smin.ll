; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘llvm.smin.*’ intrinsic
;;;
;;; Syntax:
;;;   This is an overloaded intrinsic. You can use @llvm.smin on any
;;;   integer bit width or any vector of integer elements.
;;;
;;; declare i32 @llvm.smin.i32(i32 %a, i32 %b)
;;; declare <4 x i32> @llvm.smin.v4i32(<4 x i32> %a, <4 x i32> %b)
;;;
;;; Overview:
;;;   Return the smaller of %a and %b comparing the values as signed
;;;   integers. Vector intrinsics operate on a per-element basis.
;;;   The smaller element of %a and %b at a given index is returned
;;;   for that index.
;;;
;;; Arguments:
;;;   The arguments (%a and %b) may be of any integer type or a vector
;;;   with integer element type. The argument types must match each
;;;   other, and the return type must match the argument type.
;;;
;;; Note:
;;;   We test only i8/i16/i32/i64/i128.

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i8 @func_smin_var_i8(i8 noundef signext %0, i8 noundef signext %1) {
; CHECK-LABEL: func_smin_var_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call i8 @llvm.smin.i8(i8 %0, i8 %1)
  ret i8 %3
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i16 @func_smin_var_i16(i16 noundef signext %0, i16 noundef signext %1) {
; CHECK-LABEL: func_smin_var_i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call i16 @llvm.smin.i16(i16 %0, i16 %1)
  ret i16 %3
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i32 @func_smin_var_i32(i32 noundef signext %0, i32 noundef signext %1) {
; CHECK-LABEL: func_smin_var_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call i32 @llvm.smin.i32(i32 %0, i32 %1)
  ret i32 %3
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define i64 @func_smin_var_i64(i64 noundef %0, i64 noundef %1) {
; CHECK-LABEL: func_smin_var_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.l %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call i64 @llvm.smin.i64(i64 %0, i64 %1)
  ret i64 %3
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define i128 @func_smin_var_i128(i128 noundef %0, i128 noundef %1) {
; CHECK-LABEL: func_smin_var_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s5, %s1, %s3
; CHECK-NEXT:    or %s4, 0, %s2
; CHECK-NEXT:    cmov.l.lt %s4, %s0, %s5
; CHECK-NEXT:    cmpu.l %s6, %s0, %s2
; CHECK-NEXT:    cmov.l.lt %s2, %s0, %s6
; CHECK-NEXT:    cmov.l.eq %s4, %s2, %s5
; CHECK-NEXT:    mins.l %s1, %s1, %s3
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call i128 @llvm.smin.i128(i128 %0, i128 %1)
  ret i128 %3
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i8 @func_smin_fore_zero_i8(i8 noundef signext %0) {
; CHECK-LABEL: func_smin_fore_zero_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, 0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i8 @llvm.smin.i8(i8 %0, i8 0)
  ret i8 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i16 @func_smin_fore_zero_i16(i16 noundef signext %0) {
; CHECK-LABEL: func_smin_fore_zero_i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, 0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i16 @llvm.smin.i16(i16 %0, i16 0)
  ret i16 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i32 @func_smin_fore_zero_i32(i32 noundef signext %0) {
; CHECK-LABEL: func_smin_fore_zero_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, 0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i32 @llvm.smin.i32(i32 %0, i32 0)
  ret i32 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define i64 @func_smin_fore_zero_i64(i64 noundef %0) {
; CHECK-LABEL: func_smin_fore_zero_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.l %s0, 0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.smin.i64(i64 %0, i64 0)
  ret i64 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define i128 @func_smin_fore_zero_i128(i128 noundef %0) {
; CHECK-LABEL: func_smin_fore_zero_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmps.l %s3, %s1, (0)1
; CHECK-NEXT:    sra.l %s4, %s1, 63
; CHECK-NEXT:    and %s0, %s4, %s0
; CHECK-NEXT:    cmov.l.eq %s0, %s2, %s3
; CHECK-NEXT:    mins.l %s1, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i128 @llvm.smin.i128(i128 %0, i128 0)
  ret i128 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i8 @func_smin_back_zero_i8(i8 noundef signext %0) {
; CHECK-LABEL: func_smin_back_zero_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, 0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i8 @llvm.smin.i8(i8 %0, i8 0)
  ret i8 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i16 @func_smin_back_zero_i16(i16 noundef signext %0) {
; CHECK-LABEL: func_smin_back_zero_i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, 0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i16 @llvm.smin.i16(i16 %0, i16 0)
  ret i16 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i32 @func_smin_back_zero_i32(i32 noundef signext %0) {
; CHECK-LABEL: func_smin_back_zero_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, 0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i32 @llvm.smin.i32(i32 %0, i32 0)
  ret i32 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define i64 @func_smin_back_zero_i64(i64 noundef %0) {
; CHECK-LABEL: func_smin_back_zero_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.l %s0, 0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.smin.i64(i64 %0, i64 0)
  ret i64 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define i128 @func_smin_back_zero_i128(i128 noundef %0) {
; CHECK-LABEL: func_smin_back_zero_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmps.l %s3, %s1, (0)1
; CHECK-NEXT:    sra.l %s4, %s1, 63
; CHECK-NEXT:    and %s0, %s4, %s0
; CHECK-NEXT:    cmov.l.eq %s0, %s2, %s3
; CHECK-NEXT:    mins.l %s1, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i128 @llvm.smin.i128(i128 %0, i128 0)
  ret i128 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i8 @func_smin_fore_const_i8(i8 noundef signext %0) {
; CHECK-LABEL: func_smin_fore_const_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, -1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i8 @llvm.smin.i8(i8 %0, i8 -1)
  ret i8 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i16 @func_smin_fore_const_i16(i16 noundef signext %0) {
; CHECK-LABEL: func_smin_fore_const_i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, %s0, (56)0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i16 @llvm.smin.i16(i16 %0, i16 255)
  ret i16 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i32 @func_smin_fore_const_i32(i32 noundef signext %0) {
; CHECK-LABEL: func_smin_fore_const_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, %s0, (56)0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i32 @llvm.smin.i32(i32 %0, i32 255)
  ret i32 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define i64 @func_smin_fore_const_i64(i64 noundef %0) {
; CHECK-LABEL: func_smin_fore_const_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.l %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.smin.i64(i64 %0, i64 255)
  ret i64 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define i128 @func_smin_fore_const_i128(i128 noundef %0) {
; CHECK-LABEL: func_smin_fore_const_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s3, %s1, (0)1
; CHECK-NEXT:    lea %s4, 255
; CHECK-NEXT:    lea %s2, 255
; CHECK-NEXT:    cmov.l.lt %s2, %s0, %s3
; CHECK-NEXT:    cmpu.l %s5, %s0, (56)0
; CHECK-NEXT:    cmov.l.lt %s4, %s0, %s5
; CHECK-NEXT:    cmov.l.eq %s2, %s4, %s3
; CHECK-NEXT:    mins.l %s1, 0, %s1
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i128 @llvm.smin.i128(i128 %0, i128 255)
  ret i128 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i8 @func_smin_back_const_i8(i8 noundef signext %0) {
; CHECK-LABEL: func_smin_back_const_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, -1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i8 @llvm.smin.i8(i8 %0, i8 -1)
  ret i8 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i16 @func_smin_back_const_i16(i16 noundef signext %0) {
; CHECK-LABEL: func_smin_back_const_i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, %s0, (56)0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i16 @llvm.smin.i16(i16 %0, i16 255)
  ret i16 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define signext i32 @func_smin_back_const_i32(i32 noundef signext %0) {
; CHECK-LABEL: func_smin_back_const_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.w.sx %s0, %s0, (56)0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i32 @llvm.smin.i32(i32 %0, i32 255)
  ret i32 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define i64 @func_smin_back_const_i64(i64 noundef %0) {
; CHECK-LABEL: func_smin_back_const_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    mins.l %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i64 @llvm.smin.i64(i64 %0, i64 255)
  ret i64 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define i128 @func_smin_back_const_i128(i128 noundef %0) {
; CHECK-LABEL: func_smin_back_const_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s3, %s1, (0)1
; CHECK-NEXT:    lea %s4, 255
; CHECK-NEXT:    lea %s2, 255
; CHECK-NEXT:    cmov.l.lt %s2, %s0, %s3
; CHECK-NEXT:    cmpu.l %s5, %s0, (56)0
; CHECK-NEXT:    cmov.l.lt %s4, %s0, %s5
; CHECK-NEXT:    cmov.l.eq %s2, %s4, %s3
; CHECK-NEXT:    mins.l %s1, 0, %s1
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call i128 @llvm.smin.i128(i128 %0, i128 255)
  ret i128 %2
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.smin.i32(i32, i32)

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i8 @llvm.smin.i8(i8, i8)

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i16 @llvm.smin.i16(i16, i16)

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i64 @llvm.smin.i64(i64, i64)

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i128 @llvm.smin.i128(i128, i128)
