; Test LOCFH.  See comments in asm-18.ll about testing high-word operations.
;
; RUN: llc < %s -verify-machineinstrs -mtriple=s390x-linux-gnu -mcpu=z13 \
; RUN:   -no-integrated-as | FileCheck -allow-deprecated-dag-overlap %s

declare void @foo(ptr)

; Test the simple case.
define void @f1(ptr %ptr, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r3, 42
; CHECK: locfhhe [[REG]], 0(%r2)
; CHECK: br %r14
  %easy = call i32 asm "stepa $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  %other = load i32, ptr %ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  call void asm sideeffect "stepb $0", "h"(i32 %res)
  ret void
}

; ...and again with the operands swapped.
define void @f2(ptr %ptr, i32 %limit) {
; CHECK-LABEL: f2:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r3, 42
; CHECK: locfhl [[REG]], 0(%r2)
; CHECK: br %r14
  %easy = call i32 asm "stepa $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  %other = load i32, ptr %ptr
  %res = select i1 %cond, i32 %other, i32 %easy
  call void asm sideeffect "stepb $0", "h"(i32 %res)
  ret void
}

; Check the high end of the aligned LOC range.
define void @f3(ptr %base, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r3, 42
; CHECK: locfhhe [[REG]], 524284(%r2)
; CHECK: br %r14
  %easy = call i32 asm "stepa $0", "=h"()
  %ptr = getelementptr i32, ptr %base, i64 131071
  %cond = icmp ult i32 %limit, 42
  %other = load i32, ptr %ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  call void asm sideeffect "stepb $0", "h"(i32 %res)
  ret void
}

; Check the next word up.  Other sequences besides this one would be OK.
define void @f4(ptr %base, i32 %limit) {
; CHECK-LABEL: f4:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: agfi %r2, 524288
; CHECK-DAG: clfi %r3, 42
; CHECK: locfhhe [[REG]], 0(%r2)
; CHECK: br %r14
  %easy = call i32 asm "stepa $0", "=h"()
  %ptr = getelementptr i32, ptr %base, i64 131072
  %cond = icmp ult i32 %limit, 42
  %other = load i32, ptr %ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  call void asm sideeffect "stepb $0", "h"(i32 %res)
  ret void
}

; Check the low end of the LOC range.
define void @f5(ptr %base, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r3, 42
; CHECK: locfhhe [[REG]], -524288(%r2)
; CHECK: br %r14
  %easy = call i32 asm "stepa $0", "=h"()
  %ptr = getelementptr i32, ptr %base, i64 -131072
  %cond = icmp ult i32 %limit, 42
  %other = load i32, ptr %ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  call void asm sideeffect "stepb $0", "h"(i32 %res)
  ret void
}

; Check the next word down, with the same comments as f4.
define void @f6(ptr %base, i32 %limit) {
; CHECK-LABEL: f6:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r3, 42
; CHECK-DAG: agfi %r2, -524292
; CHECK-DAG: clfi %r3, 42
; CHECK: locfhhe [[REG]], 0(%r2)
; CHECK: br %r14
  %easy = call i32 asm "stepa $0", "=h"()
  %ptr = getelementptr i32, ptr %base, i64 -131073
  %cond = icmp ult i32 %limit, 42
  %other = load i32, ptr %ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  call void asm sideeffect "stepb $0", "h"(i32 %res)
  ret void
}

; Try a frame index base.
define void @f7(i32 %alt, i32 %limit) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: locfhhe [[REG]], {{[0-9]+}}(%r15)
; CHECK: br %r14
  %ptr = alloca i32
  call void @foo(ptr %ptr)
  %easy = call i32 asm "stepa $0", "=h"()
  %cond = icmp ult i32 %limit, 42
  %other = load i32, ptr %ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  call void asm sideeffect "stepb $0", "h"(i32 %res)
  ret void
}

; Try a case when an index is involved.
define void @f8(i32 %limit, i64 %base, i64 %index) {
; CHECK-LABEL: f8:
; CHECK-DAG: stepa [[REG:%r[0-5]]]
; CHECK-DAG: clfi %r2, 42
; CHECK: locfhhe [[REG]], 0({{%r[1-5]}})
; CHECK: br %r14
  %easy = call i32 asm "stepa $0", "=h"()
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to ptr
  %cond = icmp ult i32 %limit, 42
  %other = load i32, ptr %ptr
  %res = select i1 %cond, i32 %easy, i32 %other
  call void asm sideeffect "stepb $0", "h"(i32 %res)
  ret void
}

; Test that conditionally-executed loads do not use LOC, since it is allowed
; to trap even when the condition is false.
define void @f9(i32 %limit, ptr %ptr) {
; CHECK-LABEL: f9:
; CHECK-NOT: loc
; CHECK: lfh
; CHECK: br %r14
entry:
  %easy = call i32 asm "stepa $0", "=h"()
  %cmp = icmp ule i32 %easy, %limit
  br i1 %cmp, label %load, label %exit

load:
  %other = load i32, ptr %ptr
  br label %exit

exit:
  %res = phi i32 [ %easy, %entry ], [ %other, %load ]
  call void asm sideeffect "stepb $0", "h"(i32 %res)
  ret void
}
