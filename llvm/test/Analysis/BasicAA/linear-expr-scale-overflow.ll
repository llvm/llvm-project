; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; A linear expression V * Scale + Offset keeps its nsw flag only if V * Scale
; cannot overflow. That does not follow from nsw on every operation when the
; scale itself overflows, or when a constant has been folded into the offset.

declare void @llvm.assume(i1)

; shl nsw x, 63 is defined for x == -1 and is INT64_MIN, but x * INT64_MIN
; overflows. For x == -1, %g1 is %p - 2^63 - 2^63, which is %p.
; CHECK-LABEL: Function: shl_sign_bit
; CHECK: MayAlias: i8* %g1, i8* %p
define void @shl_sign_bit(ptr %p, i1 %c) {
  %x = sext i1 %c to i64
  %s = shl nsw i64 %x, 63
  %g0 = getelementptr i8, ptr %p, i64 %s
  %g1 = getelementptr i8, ptr %g0, i64 -9223372036854775808
  load i8, ptr %g1
  load i8, ptr %p
  ret void
}

; x * 2^32 * 2^31 is x * 2^63: for x == -1 both multiplications are defined
; and the result is INT64_MIN, but x * INT64_MIN overflows.
; CHECK-LABEL: Function: mul_scale_overflows
; CHECK: MayAlias: i8* %g1, i8* %p
define void @mul_scale_overflows(ptr %p, i1 %c) {
  %x = sext i1 %c to i64
  %m1 = mul nsw i64 %x, 4294967296
  %m2 = mul nsw i64 %m1, 2147483648
  %g0 = getelementptr i8, ptr %p, i64 %m2
  %g1 = getelementptr i8, ptr %g0, i64 -9223372036854775808
  load i8, ptr %g1
  load i8, ptr %p
  ret void
}

; (x + 2^62) << 1 is 2*x + 2^63 without wrapping, but 2*x alone overflows for
; x == INT64_MIN, where %g is %p - 2^63, which is %q.
; CHECK-LABEL: Function: shl_of_add
; CHECK: MayAlias: i8* %g, i8* %q
define void @shl_of_add(ptr %p, i64 %x) {
  %c = icmp ne i64 %x, 0
  call void @llvm.assume(i1 %c)
  %a = add nsw i64 %x, 4611686018427387904
  %s = shl nsw i64 %a, 1
  %g = getelementptr i8, ptr %p, i64 %s
  %q = getelementptr i8, ptr %p, i64 -9223372036854775808
  load i8, ptr %g
  load i8, ptr %q
  ret void
}

; nsw is still used for a shift that does not reach the sign bit: x != 0, so
; %g is at least 8 bytes away from %p.
; CHECK-LABEL: Function: shl_nsw
; CHECK: NoAlias: i64* %g, i64* %p
define void @shl_nsw(ptr %p, i64 %x) {
  %c = icmp ne i64 %x, 0
  call void @llvm.assume(i1 %c)
  %s = shl nsw i64 %x, 3
  %g = getelementptr i8, ptr %p, i64 %s
  load i64, ptr %g
  load i64, ptr %p
  ret void
}

; And for multiplications whose combined scale does not overflow.
; CHECK-LABEL: Function: mul_nsw_chain
; CHECK: NoAlias: i64* %g, i64* %p
define void @mul_nsw_chain(ptr %p, i64 %x) {
  %c = icmp ne i64 %x, 0
  call void @llvm.assume(i1 %c)
  %m1 = mul nsw i64 %x, 3
  %m2 = mul nsw i64 %m1, 5
  %g = getelementptr i8, ptr %p, i64 %m2
  load i64, ptr %g
  load i64, ptr %p
  ret void
}
