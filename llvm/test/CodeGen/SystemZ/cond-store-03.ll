; Test 32-bit conditional stores that are presented as selects.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

declare void @foo(ptr)

; Test the simple case, with the loaded value first.
define void @f1(ptr %ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, ptr %ptr
  ret void
}

; ...and with the loaded value second
define void @f2(ptr %ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %alt, i32 %orig
  store i32 %res, ptr %ptr
  ret void
}

; Test cases where the value is explicitly sign-extended to 64 bits, with the
; loaded value first.
define void @f3(ptr %ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %ext = sext i32 %orig to i64
  %res = select i1 %cond, i64 %ext, i64 %alt
  %trunc = trunc i64 %res to i32
  store i32 %trunc, ptr %ptr
  ret void
}

; ...and with the loaded value second
define void @f4(ptr %ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %ext = sext i32 %orig to i64
  %res = select i1 %cond, i64 %alt, i64 %ext
  %trunc = trunc i64 %res to i32
  store i32 %trunc, ptr %ptr
  ret void
}

; Test cases where the value is explicitly zero-extended to 32 bits, with the
; loaded value first.
define void @f5(ptr %ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %ext = zext i32 %orig to i64
  %res = select i1 %cond, i64 %ext, i64 %alt
  %trunc = trunc i64 %res to i32
  store i32 %trunc, ptr %ptr
  ret void
}

; ...and with the loaded value second
define void @f6(ptr %ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f6:
; CHECK-NOT: %r2
; CHECK: bher %r14
; CHECK-NOT: %r2
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %ext = zext i32 %orig to i64
  %res = select i1 %cond, i64 %alt, i64 %ext
  %trunc = trunc i64 %res to i32
  store i32 %trunc, ptr %ptr
  ret void
}

; Check the high end of the aligned ST range.
define void @f7(ptr %base, i32 %alt, i32 %limit) {
; CHECK-LABEL: f7:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: st %r3, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 1023
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, ptr %ptr
  ret void
}

; Check the next word up, which should use STY instead of ST.
define void @f8(ptr %base, i32 %alt, i32 %limit) {
; CHECK-LABEL: f8:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sty %r3, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 1024
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, ptr %ptr
  ret void
}

; Check the high end of the aligned STY range.
define void @f9(ptr %base, i32 %alt, i32 %limit) {
; CHECK-LABEL: f9:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sty %r3, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 131071
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, ptr %ptr
  ret void
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f10(ptr %base, i32 %alt, i32 %limit) {
; CHECK-LABEL: f10:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, 524288
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 131072
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, ptr %ptr
  ret void
}

; Check the low end of the STY range.
define void @f11(ptr %base, i32 %alt, i32 %limit) {
; CHECK-LABEL: f11:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sty %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 -131072
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, ptr %ptr
  ret void
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f12(ptr %base, i32 %alt, i32 %limit) {
; CHECK-LABEL: f12:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: agfi %r2, -524292
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 -131073
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, ptr %ptr
  ret void
}

; Check that STY allows an index.
define void @f13(i64 %base, i64 %index, i32 %alt, i32 %limit) {
; CHECK-LABEL: f13:
; CHECK-NOT: %r2
; CHECK: blr %r14
; CHECK-NOT: %r2
; CHECK: sty %r4, 4096(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, ptr %ptr
  ret void
}

; Check that volatile loads are not matched.
define void @f14(ptr %ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f14:
; CHECK: l {{%r[0-5]}}, 0(%r2)
; CHECK: {{jl|jnl}} [[LABEL:[^ ]*]]
; CHECK: [[LABEL]]:
; CHECK: st {{%r[0-5]}}, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load volatile i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, ptr %ptr
  ret void
}

; ...likewise stores.  In this case we should have a conditional load into %r3.
define void @f15(ptr %ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f15:
; CHECK: jhe [[LABEL:[^ ]*]]
; CHECK: l %r3, 0(%r2)
; CHECK: [[LABEL]]:
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store volatile i32 %res, ptr %ptr
  ret void
}

; Check that atomic loads are not matched.  The transformation is OK for
; the "unordered" case tested here, but since we don't try to handle atomic
; operations at all in this context, it seems better to assert that than
; to restrict the test to a stronger ordering.
define void @f16(ptr %ptr, i32 %alt, i32 %limit) {
; FIXME: should use a normal load instead of CS.
; CHECK-LABEL: f16:
; CHECK: l {{%r[0-5]}}, 0(%r2)
; CHECK: {{jl|jnl}} [[LABEL:[^ ]*]]
; CHECK: [[LABEL]]:
; CHECK: st {{%r[0-5]}}, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load atomic i32, ptr %ptr unordered, align 4
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, ptr %ptr
  ret void
}

; ...likewise stores.
define void @f17(ptr %ptr, i32 %alt, i32 %limit) {
; FIXME: should use a normal store instead of CS.
; CHECK-LABEL: f17:
; CHECK: jhe [[LABEL:[^ ]*]]
; CHECK: l %r3, 0(%r2)
; CHECK: [[LABEL]]:
; CHECK: st %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store atomic i32 %res, ptr %ptr unordered, align 4
  ret void
}

; Try a frame index base.
define void @f18(i32 %alt, i32 %limit) {
; CHECK-LABEL: f18:
; CHECK: brasl %r14, foo@PLT
; CHECK-NOT: %r15
; CHECK: jl [[LABEL:[^ ]*]]
; CHECK-NOT: %r15
; CHECK: st {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: [[LABEL]]:
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %ptr = alloca i32
  call void @foo(ptr %ptr)
  %cond = icmp ult i32 %limit, 420
  %orig = load i32, ptr %ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, ptr %ptr
  call void @foo(ptr %ptr)
  ret void
}
