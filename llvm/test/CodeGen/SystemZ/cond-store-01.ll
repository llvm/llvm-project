; Test 8-bit conditional stores that are presented as selects.  The volatile
; tests require z10, which use a branch instead of a LOCR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

declare void @foo(i8 *)

; Test the simple case, with the loaded value first.
define void @f1(i8 *%ptr, i8 %alt, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store i8 %res, i8 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f2(i8 *%ptr, i8 %alt, i32 %limit) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %alt, i8 %orig
  store i8 %res, i8 *%ptr
  ret void
}

; Test cases where the value is explicitly sign-extended to 32 bits, with the
; loaded value first.
define void @f3(i8 *%ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %ext = sext i8 %orig to i32
  %res = select i1 %cond, i32 %ext, i32 %alt
  %trunc = trunc i32 %res to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f4(i8 *%ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %ext = sext i8 %orig to i32
  %res = select i1 %cond, i32 %alt, i32 %ext
  %trunc = trunc i32 %res to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Test cases where the value is explicitly zero-extended to 32 bits, with the
; loaded value first.
define void @f5(i8 *%ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %ext = zext i8 %orig to i32
  %res = select i1 %cond, i32 %ext, i32 %alt
  %trunc = trunc i32 %res to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f6(i8 *%ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f6:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %ext = zext i8 %orig to i32
  %res = select i1 %cond, i32 %alt, i32 %ext
  %trunc = trunc i32 %res to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Test cases where the value is explicitly sign-extended to 64 bits, with the
; loaded value first.
define void @f7(i8 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f7:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %ext = sext i8 %orig to i64
  %res = select i1 %cond, i64 %ext, i64 %alt
  %trunc = trunc i64 %res to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f8(i8 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f8:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %ext = sext i8 %orig to i64
  %res = select i1 %cond, i64 %alt, i64 %ext
  %trunc = trunc i64 %res to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Test cases where the value is explicitly zero-extended to 64 bits, with the
; loaded value first.
define void @f9(i8 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f9:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %ext = zext i8 %orig to i64
  %res = select i1 %cond, i64 %ext, i64 %alt
  %trunc = trunc i64 %res to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f10(i8 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f10:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %ext = zext i8 %orig to i64
  %res = select i1 %cond, i64 %alt, i64 %ext
  %trunc = trunc i64 %res to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Check the high end of the STC range.
define void @f11(i8 *%base, i8 %alt, i32 %limit) {
; CHECK-LABEL: f11:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stc %r3, 4095(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store i8 %res, i8 *%ptr
  ret void
}

; Check the next byte up, which should use STCY instead of STC.
define void @f12(i8 *%base, i8 %alt, i32 %limit) {
; CHECK-LABEL: f12:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stcy %r3, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store i8 %res, i8 *%ptr
  ret void
}

; Check the high end of the STCY range.
define void @f13(i8 *%base, i8 %alt, i32 %limit) {
; CHECK-LABEL: f13:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stcy %r3, 524287(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 524287
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store i8 %res, i8 *%ptr
  ret void
}

; Check the next byte up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f14(i8 *%base, i8 %alt, i32 %limit) {
; CHECK-LABEL: f14:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, 524288
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 524288
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store i8 %res, i8 *%ptr
  ret void
}

; Check the low end of the STCY range.
define void @f15(i8 *%base, i8 %alt, i32 %limit) {
; CHECK-LABEL: f15:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stcy %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 -524288
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store i8 %res, i8 *%ptr
  ret void
}

; Check the next byte down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f16(i8 *%base, i8 %alt, i32 %limit) {
; CHECK-LABEL: f16:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, -524289
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 -524289
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store i8 %res, i8 *%ptr
  ret void
}

; Check that STCY allows an index.
define void @f17(i64 %base, i64 %index, i8 %alt, i32 %limit) {
; CHECK-LABEL: f17:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stcy %r4, 4096(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i8 *
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store i8 %res, i8 *%ptr
  ret void
}

; Check that volatile loads are not matched.
define void @f18(i8 *%ptr, i8 %alt, i32 %limit) {
; CHECK-LABEL: f18:
; CHECK: lb {{%r[0-5]}}, 0(%r2)
; CHECK: {{jl|jnl}} [[LABEL:[^ ]*]]
; CHECK: [[LABEL]]:
; CHECK: stc {{%r[0-5]}}, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load volatile i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store i8 %res, i8 *%ptr
  ret void
}

; ...likewise stores.  In this case we should have a conditional load into %r3.
define void @f19(i8 *%ptr, i8 %alt, i32 %limit) {
; CHECK-LABEL: f19:
; CHECK: jhe [[LABEL:[^ ]*]]
; CHECK: lb %r3, 0(%r2)
; CHECK: [[LABEL]]:
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store volatile i8 %res, i8 *%ptr
  ret void
}

; Check that atomic loads are not matched.  The transformation is OK for
; the "unordered" case tested here, but since we don't try to handle atomic
; operations at all in this context, it seems better to assert that than
; to restrict the test to a stronger ordering.
define void @f20(i8 *%ptr, i8 %alt, i32 %limit) {
; FIXME: should use a normal load instead of CS.
; CHECK-LABEL: f20:
; CHECK: lb {{%r[0-9]+}}, 0(%r2)
; CHECK: {{jl|jnl}} [[LABEL:[^ ]*]]
; CHECK: [[LABEL]]:
; CHECK: stc {{%r[0-9]+}}, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load atomic i8, i8 *%ptr unordered, align 1
  %res = select i1 %cond, i8 %orig, i8 %alt
  store i8 %res, i8 *%ptr
  ret void
}

; ...likewise stores.
define void @f21(i8 *%ptr, i8 %alt, i32 %limit) {
; FIXME: should use a normal store instead of CS.
; CHECK-LABEL: f21:
; CHECK: jhe [[LABEL:[^ ]*]]
; CHECK: lb %r3, 0(%r2)
; CHECK: [[LABEL]]:
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store atomic i8 %res, i8 *%ptr unordered, align 1
  ret void
}

; Try a frame index base.
define void @f22(i8 %alt, i32 %limit) {
; CHECK-LABEL: f22:
; CHECK: brasl %r14, foo@PLT
; CHECK-NOT: %r15
; CHECK: jl [[LABEL:[^ ]*]]
; CHECK-NOT: %r15
; CHECK: stc {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: [[LABEL]]:
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %ptr = alloca i8
  call void @foo(i8 *%ptr)
  %cond = icmp ult i32 %limit, 420
  %orig = load i8, i8 *%ptr
  %res = select i1 %cond, i8 %orig, i8 %alt
  store i8 %res, i8 *%ptr
  call void @foo(i8 *%ptr)
  ret void
}
