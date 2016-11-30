; Test STOCFHs that are presented as selects.
; See comments in asm-18.ll about testing high-word operations.
;
; RUN: llc < %s -verify-machineinstrs -mtriple=s390x-linux-gnu -mcpu=z13 \
; RUN:   -no-integrated-as | FileCheck %s

declare void @foo(i32 *)

; Test the simple case, with the loaded value first.
define void @f1(i32 *%ptr, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r3, 42
; CHECK: stocfhhe [[REG]], 0(%r2)
; CHECK: br %r14
  %alt = call i32 asm "stepa $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  %orig = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  ret void
}

; ...and with the loaded value second
define void @f2(i32 *%ptr, i32 %limit) {
; CHECK-LABEL: f2:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r3, 42
; CHECK: stocfhl [[REG]], 0(%r2)
; CHECK: br %r14
  %alt = call i32 asm "stepa $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  %orig = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %alt, i32 %orig
  store i32 %res, i32 *%ptr
  ret void
}

; Check the high end of the aligned STOC range.
define void @f3(i32 *%base, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r3, 42
; CHECK: stocfhhe [[REG]], 524284(%r2)
; CHECK: br %r14
  %alt = call i32 asm "stepa $0", "=h"()
  %ptr = getelementptr i32, i32 *%base, i64 131071
  %cond = icmp ult i32 %limit, 42
  %orig = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  ret void
}

; Check the next word up.  Other sequences besides this one would be OK.
define void @f4(i32 *%base, i32 %limit) {
; CHECK-LABEL: f4:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: agfi %r2, 524288
; CHECK-DAG: clfi %r3, 42
; CHECK: stocfhhe [[REG]], 0(%r2)
; CHECK: br %r14
  %alt = call i32 asm "stepa $0", "=h"()
  %ptr = getelementptr i32, i32 *%base, i64 131072
  %cond = icmp ult i32 %limit, 42
  %orig = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  ret void
}

; Check the low end of the STOC range.
define void @f5(i32 *%base, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r3, 42
; CHECK: stocfhhe [[REG]], -524288(%r2)
; CHECK: br %r14
  %alt = call i32 asm "stepa $0", "=h"()
  %ptr = getelementptr i32, i32 *%base, i64 -131072
  %cond = icmp ult i32 %limit, 42
  %orig = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  ret void
}

; Check the next word down, with the same comments as f8.
define void @f6(i32 *%base, i32 %limit) {
; CHECK-LABEL: f6:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: agfi %r2, -524292
; CHECK-DAG: clfi %r3, 42
; CHECK: stocfhhe [[REG]], 0(%r2)
; CHECK: br %r14
  %alt = call i32 asm "stepa $0", "=h"()
  %ptr = getelementptr i32, i32 *%base, i64 -131073
  %cond = icmp ult i32 %limit, 42
  %orig = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  ret void
}

; Try a frame index base.
define void @f7(i32 %limit) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: stocfhhe [[REG]], {{[0-9]+}}(%r15)
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %ptr = alloca i32
  call void @foo(i32 *%ptr)
  %alt = call i32 asm "stepa $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  %orig = load i32, i32 *%ptr
  %res = select i1 %cond, i32 %orig, i32 %alt
  store i32 %res, i32 *%ptr
  call void @foo(i32 *%ptr)
  ret void
}

; Test that conditionally-executed stores do not use STOC, since STOC
; is allowed to trap even when the condition is false.
define void @f8(i32 %a, i32 %b, i32 *%dest) {
; CHECK-LABEL: f8:
; CHECK-NOT: stoc
; CHECK: stfh
; CHECK: br %r14
entry:
  %val = call i32 asm "stepa $0", "=h"()
  %cmp = icmp ule i32 %a, %b
  br i1 %cmp, label %store, label %exit

store:
  store i32 %val, i32 *%dest
  br label %exit

exit:
  ret void
}
