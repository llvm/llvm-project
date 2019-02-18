; Test LOCG.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i64 @foo(i64 *)

; Test the simple case.
define i64 @f1(i64 %easy, i64 *%ptr, i64 %limit) {
; CHECK-LABEL: f1:
; CHECK: clgfi %r4, 42
; CHECK: locghe %r2, 0(%r3)
; CHECK: br %r14
  %cond = icmp ult i64 %limit, 42
  %other = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %easy, i64 %other
  ret i64 %res
}

; ...and again with the operands swapped.
define i64 @f2(i64 %easy, i64 *%ptr, i64 %limit) {
; CHECK-LABEL: f2:
; CHECK: clgfi %r4, 42
; CHECK: locgl %r2, 0(%r3)
; CHECK: br %r14
  %cond = icmp ult i64 %limit, 42
  %other = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %other, i64 %easy
  ret i64 %res
}

; Check the high end of the aligned LOCG range.
define i64 @f3(i64 %easy, i64 *%base, i64 %limit) {
; CHECK-LABEL: f3:
; CHECK: clgfi %r4, 42
; CHECK: locghe %r2, 524280(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 65535
  %cond = icmp ult i64 %limit, 42
  %other = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %easy, i64 %other
  ret i64 %res
}

; Check the next doubleword up.  Other sequences besides this one would be OK.
define i64 @f4(i64 %easy, i64 *%base, i64 %limit) {
; CHECK-LABEL: f4:
; CHECK: agfi %r3, 524288
; CHECK: clgfi %r4, 42
; CHECK: locghe %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 65536
  %cond = icmp ult i64 %limit, 42
  %other = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %easy, i64 %other
  ret i64 %res
}

; Check the low end of the LOCG range.
define i64 @f5(i64 %easy, i64 *%base, i64 %limit) {
; CHECK-LABEL: f5:
; CHECK: clgfi %r4, 42
; CHECK: locghe %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 -65536
  %cond = icmp ult i64 %limit, 42
  %other = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %easy, i64 %other
  ret i64 %res
}

; Check the next doubleword down, with the same comments as f4.
define i64 @f6(i64 %easy, i64 *%base, i64 %limit) {
; CHECK-LABEL: f6:
; CHECK: agfi %r3, -524296
; CHECK: clgfi %r4, 42
; CHECK: locghe %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%base, i64 -65537
  %cond = icmp ult i64 %limit, 42
  %other = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %easy, i64 %other
  ret i64 %res
}

; Try a frame index base.
define i64 @f7(i64 %alt, i64 %limit) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK: locghe %r2, {{[0-9]+}}(%r15)
; CHECK: br %r14
  %ptr = alloca i64
  %easy = call i64 @foo(i64 *%ptr)
  %cond = icmp ult i64 %limit, 42
  %other = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %easy, i64 %other
  ret i64 %res
}

; Try a case when an index is involved.
define i64 @f8(i64 %easy, i64 %limit, i64 %base, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: clgfi %r3, 42
; CHECK: locghe %r2, 0({{%r[1-5]}})
; CHECK: br %r14
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to i64 *
  %cond = icmp ult i64 %limit, 42
  %other = load i64, i64 *%ptr
  %res = select i1 %cond, i64 %easy, i64 %other
  ret i64 %res
}

; Test that conditionally-executed loads do not use LOCG, since it is allowed
; to trap even when the condition is false.
define i64 @f9(i64 %easy, i64 %limit, i64 *%ptr) {
; CHECK-LABEL: f9:
; CHECK-NOT: locg
; CHECK: br %r14
entry:
  %cmp = icmp ule i64 %easy, %limit
  br i1 %cmp, label %load, label %exit

load:
  %other = load i64, i64 *%ptr
  br label %exit

exit:
  %res = phi i64 [ %easy, %entry ], [ %other, %load ]
  ret i64 %res
}

; Test that volatile loads do not use LOCG, since if the condition is false,
; it is unspecified whether or not the load happens.  LOCGR is fine though.
define i64 @f10(i64 %easy, i64 *%ptr, i64 %limit) {
; CHECK-LABEL: f10:
; CHECK: lg {{%r[0-9]*}}, 0(%r3)
; CHECK: locgr
; CHECK: br %r14
  %cond = icmp ult i64 %limit, 42
  %other = load volatile i64, i64 *%ptr
  %res = select i1 %cond, i64 %easy, i64 %other
  ret i64 %res
}

