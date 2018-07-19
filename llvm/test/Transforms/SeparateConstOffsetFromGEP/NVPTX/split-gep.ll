; RUN: opt < %s -separate-const-offset-from-gep -reassociate-geps-verify-no-dead-code -S | FileCheck %s

; Several unit tests for -separate-const-offset-from-gep. The transformation
; heavily relies on TargetTransformInfo, so we put these tests under
; target-specific folders.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
; target triple is necessary; otherwise TargetTransformInfo rejects any
; addressing mode.
target triple = "nvptx64-unknown-unknown"

%struct.S = type { float, double }

@struct_array = global [1024 x %struct.S] zeroinitializer, align 16
@float_2d_array = global [32 x [32 x float]] zeroinitializer, align 4

; We should not extract any struct field indices, because fields in a struct
; may have different types.
define double* @struct(i32 %i) {
entry:
  %add = add nsw i32 %i, 5
  %idxprom = sext i32 %add to i64
  %p = getelementptr inbounds [1024 x %struct.S], [1024 x %struct.S]* @struct_array, i64 0, i64 %idxprom, i32 1
  ret double* %p
}
; CHECK-LABEL: @struct(
; CHECK: getelementptr [1024 x %struct.S], [1024 x %struct.S]* @struct_array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i32 1

; We should be able to trace into sext(a + b) if a + b is non-negative
; (e.g., used as an index of an inbounds GEP) and one of a and b is
; non-negative.
define float* @sext_add(i32 %i, i32 %j) {
entry:
  %0 = add i32 %i, 1
  %1 = sext i32 %0 to i64  ; inbound sext(i + 1) = sext(i) + 1
  %2 = add i32 %j, -2
  ; However, inbound sext(j + -2) != sext(j) + -2, e.g., j = INT_MIN
  %3 = sext i32 %2 to i64
  %p = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %1, i64 %3
  ret float* %p
}
; CHECK-LABEL: @sext_add(
; CHECK-NOT: = add
; CHECK: add i32 %j, -2
; CHECK: sext
; CHECK: getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; CHECK: getelementptr inbounds float, float* %{{[a-zA-Z0-9]+}}, i64 32

; We should be able to trace into sext/zext if it can be distributed to both
; operands, e.g., sext (add nsw a, b) == add nsw (sext a), (sext b)
;
; This test verifies we can transform
;   gep base, a + sext(b +nsw 1), c + zext(d +nuw 1)
; to
;   gep base, a + sext(b), c + zext(d); gep ..., 1 * 32 + 1
define float* @ext_add_no_overflow(i64 %a, i32 %b, i64 %c, i32 %d) {
  %b1 = add nsw i32 %b, 1
  %b2 = sext i32 %b1 to i64
  %i = add i64 %a, %b2       ; i = a + sext(b +nsw 1)
  %d1 = add nuw i32 %d, 1
  %d2 = zext i32 %d1 to i64
  %j = add i64 %c, %d2       ; j = c + zext(d +nuw 1)
  %p = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %i, i64 %j
  ret float* %p
}
; CHECK-LABEL: @ext_add_no_overflow(
; CHECK: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; CHECK: getelementptr inbounds float, float* [[BASE_PTR]], i64 33

; Verifies we handle nested sext/zext correctly.
define void @sext_zext(i32 %a, i32 %b, float** %out1, float** %out2) {
entry:
  %0 = add nsw nuw i32 %a, 1
  %1 = sext i32 %0 to i48
  %2 = zext i48 %1 to i64    ; zext(sext(a +nsw nuw 1)) = zext(sext(a)) + 1
  %3 = add nsw i32 %b, 2
  %4 = sext i32 %3 to i48
  %5 = zext i48 %4 to i64    ; zext(sext(b +nsw 2)) != zext(sext(b)) + 2
  %p1 = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %2, i64 %5
  store float* %p1, float** %out1
  %6 = add nuw i32 %a, 3
  %7 = zext i32 %6 to i48
  %8 = sext i48 %7 to i64 ; sext(zext(a +nuw 3)) = zext(a +nuw 3) = zext(a) + 3
  %9 = add nsw i32 %b, 4
  %10 = zext i32 %9 to i48
  %11 = sext i48 %10 to i64  ; sext(zext(b +nsw 4)) != zext(b) + 4
  %p2 = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %8, i64 %11
  store float* %p2, float** %out2
  ret void
}
; CHECK-LABEL: @sext_zext(
; CHECK: [[BASE_PTR_1:%[a-zA-Z0-9]+]] = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; CHECK: getelementptr float, float* [[BASE_PTR_1]], i64 32
; CHECK: [[BASE_PTR_2:%[a-zA-Z0-9]+]] = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; CHECK: getelementptr float, float* [[BASE_PTR_2]], i64 96

; Similar to @ext_add_no_overflow, we should be able to trace into s/zext if
; its operand is an OR and the two operands of the OR have no common bits.
define float* @sext_or(i64 %a, i32 %b) {
entry:
  %b1 = shl i32 %b, 2
  %b2 = or i32 %b1, 1 ; (b << 2) and 1 have no common bits
  %b3 = or i32 %b1, 4 ; (b << 2) and 4 may have common bits
  %b2.ext = zext i32 %b2 to i64
  %b3.ext = sext i32 %b3 to i64
  %i = add i64 %a, %b2.ext
  %j = add i64 %a, %b3.ext
  %p = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %i, i64 %j
  ret float* %p
}
; CHECK-LABEL: @sext_or(
; CHECK: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; CHECK: getelementptr inbounds float, float* [[BASE_PTR]], i64 32

; The subexpression (b + 5) is used in both "i = a + (b + 5)" and "*out = b +
; 5". When extracting the constant offset 5, make sure "*out = b + 5" isn't
; affected.
define float* @expr(i64 %a, i64 %b, i64* %out) {
entry:
  %b5 = add i64 %b, 5
  %i = add i64 %b5, %a
  %p = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %i, i64 0
  store i64 %b5, i64* %out
  ret float* %p
}
; CHECK-LABEL: @expr(
; CHECK: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 0
; CHECK: getelementptr inbounds float, float* [[BASE_PTR]], i64 160
; CHECK: store i64 %b5, i64* %out

; d + sext(a +nsw (b +nsw (c +nsw 8))) => (d + sext(a) + sext(b) + sext(c)) + 8
define float* @sext_expr(i32 %a, i32 %b, i32 %c, i64 %d) {
entry:
  %0 = add nsw i32 %c, 8
  %1 = add nsw i32 %b, %0
  %2 = add nsw i32 %a, %1
  %3 = sext i32 %2 to i64
  %i = add i64 %d, %3
  %p = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 0, i64 %i
  ret float* %p
}
; CHECK-LABEL: @sext_expr(
; CHECK: sext i32
; CHECK: sext i32
; CHECK: sext i32
; CHECK: getelementptr inbounds float, float* %{{[a-zA-Z0-9]+}}, i64 8

; Verifies we handle "sub" correctly.
define float* @sub(i64 %i, i64 %j) {
  %i2 = sub i64 %i, 5 ; i - 5
  %j2 = sub i64 5, %j ; 5 - i
  %p = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %i2, i64 %j2
  ret float* %p
}
; CHECK-LABEL: @sub(
; CHECK: %[[j2:[a-zA-Z0-9]+]] = sub i64 0, %j
; CHECK: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 %i, i64 %[[j2]]
; CHECK: getelementptr inbounds float, float* [[BASE_PTR]], i64 -155

%struct.Packed = type <{ [3 x i32], [8 x i64] }> ; <> means packed

; Verifies we can emit correct uglygep if the address is not natually aligned.
define i64* @packed_struct(i32 %i, i32 %j) {
entry:
  %s = alloca [1024 x %struct.Packed], align 16
  %add = add nsw i32 %j, 3
  %idxprom = sext i32 %add to i64
  %add1 = add nsw i32 %i, 1
  %idxprom2 = sext i32 %add1 to i64
  %arrayidx3 = getelementptr inbounds [1024 x %struct.Packed], [1024 x %struct.Packed]* %s, i64 0, i64 %idxprom2, i32 1, i64 %idxprom
  ret i64* %arrayidx3
}
; CHECK-LABEL: @packed_struct(
; CHECK: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [1024 x %struct.Packed], [1024 x %struct.Packed]* %s, i64 0, i64 %{{[a-zA-Z0-9]+}}, i32 1, i64 %{{[a-zA-Z0-9]+}}
; CHECK: [[CASTED_PTR:%[a-zA-Z0-9]+]] = bitcast i64* [[BASE_PTR]] to i8*
; CHECK: %uglygep = getelementptr inbounds i8, i8* [[CASTED_PTR]], i64 100
; CHECK: bitcast i8* %uglygep to i64*

; We shouldn't be able to extract the 8 from "zext(a +nuw (b + 8))",
; because "zext(b + 8) != zext(b) + 8"
define float* @zext_expr(i32 %a, i32 %b) {
entry:
  %0 = add i32 %b, 8
  %1 = add nuw i32 %a, %0
  %i = zext i32 %1 to i64
  %p = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 0, i64 %i
  ret float* %p
}
; CHECK-LABEL: zext_expr(
; CHECK: getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 0, i64 %i

; Per http://llvm.org/docs/LangRef.html#id181, the indices of a off-bound gep
; should be considered sign-extended to the pointer size. Therefore,
;   gep base, (add i32 a, b) != gep (gep base, i32 a), i32 b
; because
;   sext(a + b) != sext(a) + sext(b)
;
; This test verifies we do not illegitimately extract the 8 from
;   gep base, (i32 a + 8)
define float* @i32_add(i32 %a) {
entry:
  %i = add i32 %a, 8
  %p = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 0, i32 %i
  ret float* %p
}
; CHECK-LABEL: @i32_add(
; CHECK: getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 0, i64 %{{[a-zA-Z0-9]+}}
; CHECK-NOT: getelementptr

; Verifies that we compute the correct constant offset when the index is
; sign-extended and then zero-extended. The old version of our code failed to
; handle this case because it simply computed the constant offset as the
; sign-extended value of the constant part of the GEP index.
define float* @apint(i1 %a) {
entry:
  %0 = add nsw nuw i1 %a, 1
  %1 = sext i1 %0 to i4
  %2 = zext i4 %1 to i64         ; zext (sext i1 1 to i4) to i64 = 15
  %p = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 0, i64 %2
  ret float* %p
}
; CHECK-LABEL: @apint(
; CHECK: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 0, i64 %{{[a-zA-Z0-9]+}}
; CHECK: getelementptr float, float* [[BASE_PTR]], i64 15

; Do not trace into binary operators other than ADD, SUB, and OR.
define float* @and(i64 %a) {
entry:
  %0 = shl i64 %a, 2
  %1 = and i64 %0, 1
  %p = getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array, i64 0, i64 0, i64 %1
  ret float* %p
}
; CHECK-LABEL: @and(
; CHECK: getelementptr [32 x [32 x float]], [32 x [32 x float]]* @float_2d_array
; CHECK-NOT: getelementptr

; The code that rebuilds an OR expression used to be buggy, and failed on this
; test.
define float* @shl_add_or(i64 %a, float* %ptr) {
; CHECK-LABEL: @shl_add_or(
entry:
  %shl = shl i64 %a, 2
  %add = add i64 %shl, 12
  %or = or i64 %add, 1
; CHECK: [[OR:%or[0-9]*]] = add i64 %shl, 1
  ; ((a << 2) + 12) and 1 have no common bits. Therefore,
  ; SeparateConstOffsetFromGEP is able to extract the 12.
  ; TODO(jingyue): We could reassociate the expression to combine 12 and 1.
  %p = getelementptr float, float* %ptr, i64 %or
; CHECK: [[PTR:%[a-zA-Z0-9]+]] = getelementptr float, float* %ptr, i64 [[OR]]
; CHECK: getelementptr float, float* [[PTR]], i64 12
  ret float* %p
; CHECK-NEXT: ret
}

; The source code used to be buggy in checking
; (AccumulativeByteOffset % ElementTypeSizeOfGEP == 0)
; where AccumulativeByteOffset is signed but ElementTypeSizeOfGEP is unsigned.
; The compiler would promote AccumulativeByteOffset to unsigned, causing
; unexpected results. For example, while -64 % (int64_t)24 != 0,
; -64 % (uint64_t)24 == 0.
%struct3 = type { i64, i32 }
%struct2 = type { %struct3, i32 }
%struct1 = type { i64, %struct2 }
%struct0 = type { i32, i32, i64*, [100 x %struct1] }
define %struct2* @sign_mod_unsign(%struct0* %ptr, i64 %idx) {
; CHECK-LABEL: @sign_mod_unsign(
entry:
  %arrayidx = add nsw i64 %idx, -2
; CHECK-NOT: add
  %ptr2 = getelementptr inbounds %struct0, %struct0* %ptr, i64 0, i32 3, i64 %arrayidx, i32 1
; CHECK: [[PTR:%[a-zA-Z0-9]+]] = getelementptr %struct0, %struct0* %ptr, i64 0, i32 3, i64 %idx, i32 1
; CHECK: [[PTR1:%[a-zA-Z0-9]+]] = bitcast %struct2* [[PTR]] to i8*
; CHECK: getelementptr inbounds i8, i8* [[PTR1]], i64 -64
; CHECK: bitcast
  ret %struct2* %ptr2
; CHECK-NEXT: ret
}
