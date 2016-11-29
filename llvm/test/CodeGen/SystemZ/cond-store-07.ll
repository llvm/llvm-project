; Test STOCs that are presented as selects.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Run the test again to make sure it still works the same even
; in the presence of the load-store-on-condition-2 facility.
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare void @foo(i32 *)

; Test the simple case, with the loaded value first.
define void @f1(i32 *%ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK: clfi %r4, 42
; CHECK: stoche %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %orig = load i32 , i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f2(i32 *%ptr, i32 %alt, i32 %limit) {
; CHECK-LABEL: f2:
; CHECK: clfi %r4, 42
; CHECK: stocl %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %orig = load i32 , i32 *%ptr
  %res = select i1 %cond, i32 %alt, i32 %orig
  store i32 %res, i32 *%ptr
  ret void
}

; Test cases where the value is explicitly sign-extended to 64 bits, with the
; loaded value first.
define void @f3(i32 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK: clfi %r4, 42
; CHECK: stoche %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %orig = load i32 , i32 *%ptr
  %ext = sext i32 %orig to i64
  %res = select i1 %cond, i64 %ext, i64 %alt
  %trunc = trunc i64 %res to i32
  store i32 %trunc, i32 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f4(i32 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f4:
; CHECK: clfi %r4, 42
; CHECK: stocl %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %orig = load i32 , i32 *%ptr
  %ext = sext i32 %orig to i64
  %res = select i1 %cond, i64 %alt, i64 %ext
  %trunc = trunc i64 %res to i32
  store i32 %trunc, i32 *%ptr
  ret void
}

; Test cases where the value is explicitly zero-extended to 32 bits, with the
; loaded value first.
define void @f5(i32 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK: clfi %r4, 42
; CHECK: stoche %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %orig = load i32 , i32 *%ptr
  %ext = zext i32 %orig to i64
  %res = select i1 %cond, i64 %ext, i64 %alt
  %trunc = trunc i64 %res to i32
  store i32 %trunc, i32 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f6(i32 *%ptr, i64 %alt, i32 %limit) {
; CHECK-LABEL: f6:
; CHECK: clfi %r4, 42
; CHECK: stocl %r3, 0(%r2)
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %orig = load i32 , i32 *%ptr
  %ext = zext i32 %orig to i64
  %res = select i1 %cond, i64 %alt, i64 %ext
  %trunc = trunc i64 %res to i32
  store i32 %trunc, i32 *%ptr
  ret void
}

; Check the high end of the aligned STOC range.
define void @f7(i32 *%base, i32 %alt, i32 %limit) {
; CHECK-LABEL: f7:
; CHECK: clfi %r4, 42
; CHECK: stoche %r3, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 131071
  %cond = icmp ult i32 %limit, 42
  %orig = load i32 , i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  ret void
}

; Check the next word up.  Other sequences besides this one would be OK.
define void @f8(i32 *%base, i32 %alt, i32 %limit) {
; CHECK-LABEL: f8:
; CHECK: agfi %r2, 524288
; CHECK: clfi %r4, 42
; CHECK: stoche %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 131072
  %cond = icmp ult i32 %limit, 42
  %orig = load i32 , i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  ret void
}

; Check the low end of the STOC range.
define void @f9(i32 *%base, i32 %alt, i32 %limit) {
; CHECK-LABEL: f9:
; CHECK: clfi %r4, 42
; CHECK: stoche %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 -131072
  %cond = icmp ult i32 %limit, 42
  %orig = load i32 , i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  ret void
}

; Check the next word down, with the same comments as f8.
define void @f10(i32 *%base, i32 %alt, i32 %limit) {
; CHECK-LABEL: f10:
; CHECK: agfi %r2, -524292
; CHECK: clfi %r4, 42
; CHECK: stoche %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 -131073
  %cond = icmp ult i32 %limit, 42
  %orig = load i32 , i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  ret void
}

; Try a frame index base.
define void @f11(i32 %alt, i32 %limit) {
; CHECK-LABEL: f11:
; CHECK: brasl %r14, foo@PLT
; CHECK: stoche {{%r[0-9]+}}, {{[0-9]+}}(%r15)
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %ptr = alloca i32
  call void @foo(i32 *%ptr)
  %cond = icmp ult i32 %limit, 42
  %orig = load i32 , i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  call void @foo(i32 *%ptr)
  ret void
}

; Test that conditionally-executed stores do not use STOC, since STOC
; is allowed to trap even when the condition is false.
define void @f12(i32 %a, i32 %b, i32 *%dest) {
; CHECK-LABEL: f12:
; CHECK-NOT: stoc
; CHECK: br %r14
entry:
  %cmp = icmp ule i32 %a, %b
  br i1 %cmp, label %store, label %exit

store:
  store i32 %b, i32 *%dest
  br label %exit

exit:
  ret void
}
