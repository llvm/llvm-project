; Test STOCGs that are presented as selects.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare void @foo(ptr)

; Test with the loaded value first.
define void @f1(ptr %ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK: clfi %r4, 42
; CHECK: stocghe %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %orig = load i64, ptr %ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, ptr %ptr
  ret void
}

; ...and with the loaded value second
define void @f2(ptr %ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f2:
; CHECK: clfi %r4, 42
; CHECK: stocgl %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %orig = load i64, ptr %ptr
  %res = select i1 %cond, i64 %alt, i64 %orig
  store i64 %res, ptr %ptr
  ret void
}

; Check the high end of the aligned STOCG range.
define void @f3(ptr %base, i64 %alt, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK: clfi %r4, 42
; CHECK: stocghe %r3, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i64 65535
  %cond = icmp ult i32 %limit, 42
  %orig = load i64, ptr %ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, ptr %ptr
  ret void
}

; Check the next doubleword up.  Other sequences besides this one would be OK.
define void @f4(ptr %base, i64 %alt, i32 %limit) {
; CHECK-LABEL: f4:
; CHECK: agfi %r2, 524288
; CHECK: clfi %r4, 42
; CHECK: stocghe %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i64 65536
  %cond = icmp ult i32 %limit, 42
  %orig = load i64, ptr %ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, ptr %ptr
  ret void
}

; Check the low end of the STOCG range.
define void @f5(ptr %base, i64 %alt, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK: clfi %r4, 42
; CHECK: stocghe %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i64 -65536
  %cond = icmp ult i32 %limit, 42
  %orig = load i64, ptr %ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, ptr %ptr
  ret void
}

; Check the next doubleword down, with the same comments as f4.
define void @f6(ptr %base, i64 %alt, i32 %limit) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, -524296
; CHECK: clfi %r4, 42
; CHECK: stocghe %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %base, i64 -65537
  %cond = icmp ult i32 %limit, 42
  %orig = load i64, ptr %ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, ptr %ptr
  ret void
}

; Try a frame index base.
define void @f7(i64 %alt, i32 %limit) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK: stocghe {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %ptr = alloca i64
  call void @foo(ptr %ptr)
  %cond = icmp ult i32 %limit, 42
  %orig = load i64, ptr %ptr
  %res = select i1 %cond, i64 %orig, i64 %alt
  store i64 %res, ptr %ptr
  call void @foo(ptr %ptr)
  ret void
}

; Test that conditionally-executed stores do not use STOC, since STOC
; is allowed to trap even when the condition is false.
define void @f8(i64 %a, i64 %b, ptr %dest) {
; CHECK-LABEL: f8:
; CHECK-NOT: stocg %r3, 0(%r4)
; CHECK: br %r14
entry:
  %cmp = icmp ule i64 %a, %b
  br i1 %cmp, label %store, label %exit

store:
  store i64 %b, ptr %dest
  br label %exit

exit:
  ret void
}
