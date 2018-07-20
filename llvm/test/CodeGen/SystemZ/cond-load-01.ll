; Test LOC.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Run the test again to make sure it still works the same even
; in the presence of the load-store-on-condition-2 facility.
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare i32 @foo(i32 *)

; Test the simple case.
define i32 @f1(i32 %easy, i32 *%ptr, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK: clfi %r4, 42
; CHECK: loche %r2, 0(%r3)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %other = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  ret i32 %res
}

; ...and again with the operands swapped.
define i32 @f2(i32 %easy, i32 *%ptr, i32 %limit) {
; CHECK-LABEL: f2:
; CHECK: clfi %r4, 42
; CHECK: locl %r2, 0(%r3)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %other = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %other, i32 %easy
  ret i32 %res
}

; Check the high end of the aligned LOC range.
define i32 @f3(i32 %easy, i32 *%base, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK: clfi %r4, 42
; CHECK: loche %r2, 524284(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 131071
  %cond = icmp ult i32 %limit, 42
  %other = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  ret i32 %res
}

; Check the next word up.  Other sequences besides this one would be OK.
define i32 @f4(i32 %easy, i32 *%base, i32 %limit) {
; CHECK-LABEL: f4:
; CHECK: agfi %r3, 524288
; CHECK: clfi %r4, 42
; CHECK: loche %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 131072
  %cond = icmp ult i32 %limit, 42
  %other = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  ret i32 %res
}

; Check the low end of the LOC range.
define i32 @f5(i32 %easy, i32 *%base, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK: clfi %r4, 42
; CHECK: loche %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 -131072
  %cond = icmp ult i32 %limit, 42
  %other = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  ret i32 %res
}

; Check the next word down, with the same comments as f4.
define i32 @f6(i32 %easy, i32 *%base, i32 %limit) {
; CHECK-LABEL: f6:
; CHECK: agfi %r3, -524292
; CHECK: clfi %r4, 42
; CHECK: loche %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 -131073
  %cond = icmp ult i32 %limit, 42
  %other = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  ret i32 %res
}

; Try a frame index base.
define i32 @f7(i32 %alt, i32 %limit) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK: loche %r2, {{[0-9]+}}(%r15)
; CHECK: br %r14
  %ptr = alloca i32
  %easy = call i32 @foo(i32 *%ptr)
  %cond = icmp ult i32 %limit, 42
  %other = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  ret i32 %res
}

; Try a case when an index is involved.
define i32 @f8(i32 %easy, i32 %limit, i64 %base, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: clfi %r3, 42
; CHECK: loche %r2, 0({{%r[1-5]}})
; CHECK: br %r14
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to i32 *
  %cond = icmp ult i32 %limit, 42
  %other = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  ret i32 %res
}

; Test that conditionally-executed loads do not use LOC, since it is allowed
; to trap even when the condition is false.
define i32 @f9(i32 %easy, i32 %limit, i32 *%ptr) {
; CHECK-LABEL: f9:
; CHECK-NOT: loc
; CHECK: br %r14
entry:
  %cmp = icmp ule i32 %easy, %limit
  br i1 %cmp, label %load, label %exit

load:
  %other = load i32, i32 *%ptr
  br label %exit

exit:
  %res = phi i32 [ %easy, %entry ], [ %other, %load ]
  ret i32 %res
}

; Test that volatile loads do not use LOC, since if the condition is false,
; it is unspecified whether or not the load happens.  LOCR is fine though.
define i32 @f10(i32 %easy, i32 *%ptr, i32 %limit) {
; CHECK-LABEL: f10:
; CHECK: l {{%r[0-9]*}}, 0(%r3)
; CHECK: locr
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %other = load volatile i32, i32 *%ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  ret i32 %res
}

