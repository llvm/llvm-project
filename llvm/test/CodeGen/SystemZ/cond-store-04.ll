; Test 64-bit conditional stores that are presented as selects.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

declare void @foo(i64 *)

; Test with the loaded value first.
define void @f1(i64 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stg %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, i64 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f2(i64 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: stg %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %alt, i64 %orig
  store i64 %res, i64 *%ptr
  ret void
}

; Check the high end of the aligned STG range.
define void @f3(i64 *%base, i64 %alt, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stg %r3, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 65535
  %cond = icmp ult i32 %limit, 420
  %orig = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, i64 *%ptr
  ret void
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f4(i64 *%base, i64 %alt, i32 %limit) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, 524288
; CHECK: stg %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 65536
  %cond = icmp ult i32 %limit, 420
  %orig = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, i64 *%ptr
  ret void
}

; Check the low end of the STG range.
define void @f5(i64 *%base, i64 %alt, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stg %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 -65536
  %cond = icmp ult i32 %limit, 420
  %orig = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, i64 *%ptr
  ret void
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f6(i64 *%base, i64 %alt, i32 %limit) {
; CHECK-LABEL: f6:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, -524296
; CHECK: stg %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 -65537
  %cond = icmp ult i32 %limit, 420
  %orig = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, i64 *%ptr
  ret void
}

; Check that STG allows an index.
define void @f7(i64 %base, i64 %index, i64 %alt, i32 %limit) {
; CHECK-LABEL: f7:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: stg %r4, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i64 *
  %cond = icmp ult i32 %limit, 420
  %orig = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, i64 *%ptr
  ret void
}

; Check that volatile loads are not matched.
define void @f8(i64 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f8:
; CHECK: lg {{%r[0-5]}}, 0(%r2)
; CHECK: {{jl|jnl}} [[LABEL:[^ ]*]]
; CHECK: [[LABEL]]:
; CHECK: stg {{%r[0-5]}}, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load volatile i64, i64 *%ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, i64 *%ptr
  ret void
}

; ...likewise stores.  In this case we should have a conditional load into %r3.
define void @f9(i64 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f9:
; CHECK: jhe [[LABEL:[^ ]*]]
; CHECK: lg %r3, 0(%r2)
; CHECK: [[LABEL]]:
; CHECK: stg %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store volatile i64 %res, i64 *%ptr
  ret void
}

; Check that atomic loads are not matched.  The transformation is OK for
; the "unordered" case tested here, but since we don't try to handle atomic
; operations at all in this context, it seems better to assert that than
; to restrict the test to a stronger ordering.
define void @f10(i64 *%ptr, i64 %alt, i32 %limit) {
; FIXME: should use a normal load instead of CSG.
; CHECK-LABEL: f10:
; CHECK: lg {{%r[0-5]}}, 0(%r2)
; CHECK: {{jl|jnl}} [[LABEL:[^ ]*]]
; CHECK: [[LABEL]]:
; CHECK: stg {{%r[0-5]}}, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load atomic i64, i64 *%ptr unordered, align 8
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, i64 *%ptr
  ret void
}

; ...likewise stores.
define void @f11(i64 *%ptr, i64 %alt, i32 %limit) {
; FIXME: should use a normal store instead of CSG.
; CHECK-LABEL: f11:
; CHECK: jhe [[LABEL:[^ ]*]]
; CHECK: lg %r3, 0(%r2)
; CHECK: [[LABEL]]:
; CHECK: stg %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store atomic i64 %res, i64 *%ptr unordered, align 8
  ret void
}

; Try a frame index base.
define void @f12(i64 %alt, i32 %limit) {
; CHECK-LABEL: f12:
; CHECK: brasl %r14, foo@PLT
; CHECK-NOT: %r15
; CHECK: jl [[LABEL:[^ ]*]]
; CHECK-NOT: %r15
; CHECK: stg {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: [[LABEL]]:
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %ptr = alloca i64
  call void @foo(i64 *%ptr)
  %cond = icmp ult i32 %limit, 420
  %orig = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, i64 *%ptr
  call void @foo(i64 *%ptr)
  ret void
}
