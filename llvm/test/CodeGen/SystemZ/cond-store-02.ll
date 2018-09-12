; Test 16-bit conditional stores that are presented as selects.  The volatile
; tests require z10, which use a branch instead of a LOCR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

declare void @foo(i16 *)

; Test the simple case, with the loaded value first.
define void @f1(i16 *%ptr, i16 %alt, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store i16 %res, i16 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f2(i16 *%ptr, i16 %alt, i32 %limit) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %alt, i16 %orig
  store i16 %res, i16 *%ptr
  ret void
}

; Test cases where the value is explicitly sign-extended to 32 bits, with the
; loaded value first.
define void @f3(i16 *%ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %ext = sext i16 %orig to i32
  %res = select i1 %cond, i32 %ext, i32 %alt
  %trunc = trunc i32 %res to i16
  store i16 %trunc, i16 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f4(i16 *%ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %ext = sext i16 %orig to i32
  %res = select i1 %cond, i32 %alt, i32 %ext
  %trunc = trunc i32 %res to i16
  store i16 %trunc, i16 *%ptr
  ret void
}

; Test cases where the value is explicitly zero-extended to 32 bits, with the
; loaded value first.
define void @f5(i16 *%ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %ext = zext i16 %orig to i32
  %res = select i1 %cond, i32 %ext, i32 %alt
  %trunc = trunc i32 %res to i16
  store i16 %trunc, i16 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f6(i16 *%ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f6:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %ext = zext i16 %orig to i32
  %res = select i1 %cond, i32 %alt, i32 %ext
  %trunc = trunc i32 %res to i16
  store i16 %trunc, i16 *%ptr
  ret void
}

; Test cases where the value is explicitly sign-extended to 64 bits, with the
; loaded value first.
define void @f7(i16 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f7:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %ext = sext i16 %orig to i64
  %res = select i1 %cond, i64 %ext, i64 %alt
  %trunc = trunc i64 %res to i16
  store i16 %trunc, i16 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f8(i16 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f8:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %ext = sext i16 %orig to i64
  %res = select i1 %cond, i64 %alt, i64 %ext
  %trunc = trunc i64 %res to i16
  store i16 %trunc, i16 *%ptr
  ret void
}

; Test cases where the value is explicitly zero-extended to 64 bits, with the
; loaded value first.
define void @f9(i16 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f9:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %ext = zext i16 %orig to i64
  %res = select i1 %cond, i64 %ext, i64 %alt
  %trunc = trunc i64 %res to i16
  store i16 %trunc, i16 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f10(i16 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f10:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %ext = zext i16 %orig to i64
  %res = select i1 %cond, i64 %alt, i64 %ext
  %trunc = trunc i64 %res to i16
  store i16 %trunc, i16 *%ptr
  ret void
}

; Check the high end of the aligned STH range.
define void @f11(i16 *%base, i16 %alt, i32 %limit) {
; CHECK-LABEL: f11:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sth %r3, 4094(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 2047
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store i16 %res, i16 *%ptr
  ret void
}

; Check the next halfword up, which should use STHY instead of STH.
define void @f12(i16 *%base, i16 %alt, i32 %limit) {
; CHECK-LABEL: f12:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sthy %r3, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 2048
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store i16 %res, i16 *%ptr
  ret void
}

; Check the high end of the aligned STHY range.
define void @f13(i16 *%base, i16 %alt, i32 %limit) {
; CHECK-LABEL: f13:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sthy %r3, 524286(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 262143
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store i16 %res, i16 *%ptr
  ret void
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f14(i16 *%base, i16 %alt, i32 %limit) {
; CHECK-LABEL: f14:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, 524288
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 262144
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store i16 %res, i16 *%ptr
  ret void
}

; Check the low end of the STHY range.
define void @f15(i16 *%base, i16 %alt, i32 %limit) {
; CHECK-LABEL: f15:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sthy %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 -262144
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store i16 %res, i16 *%ptr
  ret void
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f16(i16 *%base, i16 %alt, i32 %limit) {
; CHECK-LABEL: f16:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, -524290
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%base, i64 -262145
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store i16 %res, i16 *%ptr
  ret void
}

; Check that STHY allows an index.
define void @f17(i64 %base, i64 %index, i16 %alt, i32 %limit) {
; CHECK-LABEL: f17:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sthy %r4, 4096(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i16 *
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store i16 %res, i16 *%ptr
  ret void
}

; Check that volatile loads are not matched.
define void @f18(i16 *%ptr, i16 %alt, i32 %limit) {
; CHECK-LABEL: f18:
; CHECK: lh {{%r[0-5]}}, 0(%r2)
; CHECK: {{jl|jnl}} [[LABEL:[^ ]*]]
; CHECK: [[LABEL]]:
; CHECK: sth {{%r[0-5]}}, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load volatile i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store i16 %res, i16 *%ptr
  ret void
}

; ...likewise stores.  In this case we should have a conditional load into %r3.
define void @f19(i16 *%ptr, i16 %alt, i32 %limit) {
; CHECK-LABEL: f19:
; CHECK: jhe [[LABEL:[^ ]*]]
; CHECK: lh %r3, 0(%r2)
; CHECK: [[LABEL]]:
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store volatile i16 %res, i16 *%ptr
  ret void
}

; Check that atomic loads are not matched.  The transformation is OK for
; the "unordered" case tested here, but since we don't try to handle atomic
; operations at all in this context, it seems better to assert that than
; to restrict the test to a stronger ordering.
define void @f20(i16 *%ptr, i16 %alt, i32 %limit) {
; FIXME: should use a normal load instead of CS.
; CHECK-LABEL: f20:
; CHECK: lh {{%r[0-9]+}}, 0(%r2)
; CHECK: {{jl|jnl}} [[LABEL:[^ ]*]]
; CHECK: [[LABEL]]:
; CHECK: sth {{%r[0-9]+}}, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load atomic i16, i16 *%ptr unordered, align 2
  %res = select i1 %cond, i16 %orig, i16 %alt
  store i16 %res, i16 *%ptr
  ret void
}

; ...likewise stores.
define void @f21(i16 *%ptr, i16 %alt, i32 %limit) {
; FIXME: should use a normal store instead of CS.
; CHECK-LABEL: f21:
; CHECK: jhe [[LABEL:[^ ]*]]
; CHECK: lh %r3, 0(%r2)
; CHECK: [[LABEL]]:
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store atomic i16 %res, i16 *%ptr unordered, align 2
  ret void
}

; Try a frame index base.
define void @f22(i16 %alt, i32 %limit) {
; CHECK-LABEL: f22:
; CHECK: brasl %r14, foo@PLT
; CHECK-NOT: %r15
; CHECK: jl [[LABEL:[^ ]*]]
; CHECK-NOT: %r15
; CHECK: sth {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: [[LABEL]]:
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %ptr = alloca i16
  call void @foo(i16 *%ptr)
  %cond = icmp ult i32 %limit, 420
  %orig = load i16, i16 *%ptr
  %res = select i1 %cond, i16 %orig, i16 %alt
  store i16 %res, i16 *%ptr
  call void @foo(i16 *%ptr)
  ret void
}
