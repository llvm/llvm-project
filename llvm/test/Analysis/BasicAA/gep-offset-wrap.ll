; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; The decomposition of two GEPs gives the distance between them as a constant
; plus scaled indices, but only modulo 2^64 unless the GEPs cannot wrap. Rules
; that treat that sum as an integer must not be used when it can wrap around.

declare void @llvm.assume(i1)

; 4 + 3*x + 3*y is 2^64 for x == y == 3074457345618258602. Neither
; multiplication overflows, and %g2 is %p.
; CHECK-LABEL: Function: gcd_two_indices_may_wrap
; CHECK: MayAlias: i8* %g2, i8* %p
define void @gcd_two_indices_may_wrap(ptr %p, i64 %x, i64 %y) {
  %mx = mul nsw i64 %x, 3
  %my = mul nsw i64 %y, 3
  %g0 = getelementptr i8, ptr %p, i64 4
  %g1 = getelementptr i8, ptr %g0, i64 %mx
  %g2 = getelementptr i8, ptr %g1, i64 %my
  load i8, ptr %g2
  load i8, ptr %p
  ret void
}

; With nusw GEPs the same sum cannot wrap.
; CHECK-LABEL: Function: gcd_two_indices_nusw
; CHECK: NoAlias: i8* %g2, i8* %p
define void @gcd_two_indices_nusw(ptr %p, i64 %x, i64 %y) {
  %mx = mul nsw i64 %x, 3
  %my = mul nsw i64 %y, 3
  %g0 = getelementptr nusw i8, ptr %p, i64 4
  %g1 = getelementptr nusw i8, ptr %g0, i64 %mx
  %g2 = getelementptr nusw i8, ptr %g1, i64 %my
  load i8, ptr %g2
  load i8, ptr %p
  ret void
}

; Indices sign extended from i32 are too small for the sum to wrap.
; CHECK-LABEL: Function: gcd_two_narrow_indices
; CHECK: NoAlias: i8* %g2, i8* %p
define void @gcd_two_narrow_indices(ptr %p, i32 %x, i32 %y) {
  %sx = sext i32 %x to i64
  %sy = sext i32 %y to i64
  %mx = mul nsw i64 %sx, 3
  %my = mul nsw i64 %sy, 3
  %g0 = getelementptr i8, ptr %p, i64 4
  %g1 = getelementptr i8, ptr %g0, i64 %mx
  %g2 = getelementptr i8, ptr %g1, i64 %my
  load i8, ptr %g2
  load i8, ptr %p
  ret void
}

; A single nsw index and a small constant cannot wrap either.
; CHECK-LABEL: Function: gcd_one_index
; CHECK: NoAlias: i8* %g1, i8* %p
define void @gcd_one_index(ptr %p, i64 %x) {
  %mx = mul nsw i64 %x, 3
  %g0 = getelementptr i8, ptr %p, i64 4
  %g1 = getelementptr i8, ptr %g0, i64 %mx
  load i8, ptr %g1
  load i8, ptr %p
  ret void
}

; The constants can wrap as well, even if every GEP is nusw: %g is %p for
; x == -1317624576693539401, and the constant difference between %g and %q is
; 2^63.
; CHECK-LABEL: Function: gcd_constant_wraps
; CHECK: MayAlias: i8* %g, i16* %q
define void @gcd_constant_wraps(ptr %p, i64 %x) {
  %mx = mul nsw i64 %x, 7
  %h = getelementptr nusw i8, ptr %p, i64 9223372036854775807
  %g = getelementptr nusw i8, ptr %h, i64 %mx
  %q = getelementptr nusw i8, ptr %p, i64 -1
  load i8, ptr %g
  load i16, ptr %q
  ret void
}

; 4 + 6*x - 6*y is 2^64 for x == 1537228672809129301 and y == -x. Both
; multiplications are nsw and x != y, and %g2 is %p.
; CHECK-LABEL: Function: minabs_two_indices_may_wrap
; CHECK: MayAlias: i8* %g2, i64* %p
define void @minabs_two_indices_may_wrap(ptr %p, i64 %x, i64 %y) {
  %c = icmp ne i64 %x, %y
  call void @llvm.assume(i1 %c)
  %mx = mul nsw i64 %x, 6
  %my = mul nsw i64 %y, -6
  %g0 = getelementptr i8, ptr %p, i64 4
  %g1 = getelementptr i8, ptr %g0, i64 %mx
  %g2 = getelementptr i8, ptr %g1, i64 %my
  load i8, ptr %g2
  load i64, ptr %p
  ret void
}

; CHECK-LABEL: Function: minabs_two_indices_nusw
; CHECK: NoAlias: i8* %g2, i64* %p
define void @minabs_two_indices_nusw(ptr %p, i64 %x, i64 %y) {
  %c = icmp ne i64 %x, %y
  call void @llvm.assume(i1 %c)
  %mx = mul nsw i64 %x, 6
  %my = mul nsw i64 %y, -6
  %g0 = getelementptr nusw i8, ptr %p, i64 4
  %g1 = getelementptr nusw i8, ptr %g0, i64 %mx
  %g2 = getelementptr nusw i8, ptr %g1, i64 %my
  load i8, ptr %g2
  load i64, ptr %p
  ret void
}

; For x == 0 the two indices are 2^63 - 2 and -(2^63 - 2). Neither
; multiplication overflows, and %g2 is %g1 + 4.
; CHECK-LABEL: Function: constant_offset_heuristic_may_wrap
; CHECK: MayAlias: i64* %g1, i64* %g2
define void @constant_offset_heuristic_may_wrap(ptr %p, i32 %x) {
  %a = add i32 %x, 1
  %b = add i32 %x, -1
  %sa = sext i32 %a to i64
  %sb = sext i32 %b to i64
  %ia = mul nsw i64 %sa, 9223372036854775806
  %ib = mul nsw i64 %sb, 9223372036854775806
  %g1 = getelementptr i8, ptr %p, i64 %ia
  %g2 = getelementptr i8, ptr %p, i64 %ib
  load i64, ptr %g1
  load i64, ptr %g2
  ret void
}
