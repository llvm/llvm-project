; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
; NOTE: A printer test checks that only real users constrain IV choices.
; RUN: opt -passes='print<iv-users>' -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare void @llvm.assume(i1)
declare void @observe(i1)

; Both assumption comparisons are ephemeral, although their shared index also
; addresses a real store. Keep that address use and the loop exit.
define void @assume_only(ptr %out, i64 %n, i64 %left, i64 %right) {
entry:
  br label %assume_loop
assume_loop:
  %i = phi i64 [ 0, %entry ], [ %next, %assume_loop ]
  %second = or disjoint i64 %i, 1
  %left_ok = icmp ult i64 %second, %left
  call void @llvm.assume(i1 %left_ok)
  %right_ok = icmp ult i64 %second, %right
  call void @llvm.assume(i1 %right_ok)
  %pointer = getelementptr i64, ptr %out, i64 %second
  store i64 %i, ptr %pointer
  %next = add nuw i64 %i, 2
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %assume_loop
exit:
  ret void
}
; CHECK-LABEL: IV Users for loop %assume_loop
; CHECK-NOT: in  %left_ok =
; CHECK-NOT: in  %right_ok =
; CHECK: in  %done = icmp eq i64 %next, %n
; CHECK-NOT: in  %left_ok =
; CHECK-NOT: in  %right_ok =
; CHECK: %pointer = {{.*}} in  store i64 %i, ptr %pointer
; CHECK-NOT: in  %left_ok =
; CHECK-NOT: in  %right_ok =

; A comparison with a real user must remain, even when an assume also uses it.
define void @observable_comparison(ptr %out, i64 %n, i64 %limit) {
entry:
  br label %observable_loop
observable_loop:
  %i = phi i64 [ 0, %entry ], [ %next, %observable_loop ]
  %second = or disjoint i64 %i, 1
  %observable_ok = icmp ult i64 %second, %limit
  call void @llvm.assume(i1 %observable_ok)
  call void @observe(i1 %observable_ok)
  %pointer = getelementptr i64, ptr %out, i64 %second
  store i64 %i, ptr %pointer
  %next = add nuw i64 %i, 2
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %observable_loop
exit:
  ret void
}
; CHECK-LABEL: IV Users for loop %observable_loop
; CHECK: in  %observable_ok = icmp ult

; Multiple assumption uses do not make an otherwise ephemeral comparison real.
define void @multiple_assumptions(ptr %out, i64 %n, i64 %limit) {
entry:
  br label %multiple_loop
multiple_loop:
  %i = phi i64 [ 0, %entry ], [ %next, %multiple_loop ]
  %second = or disjoint i64 %i, 1
  %only_assumed = icmp ult i64 %second, %limit
  call void @llvm.assume(i1 %only_assumed)
  call void @llvm.assume(i1 %only_assumed)
  %pointer = getelementptr i64, ptr %out, i64 %second
  store i64 %i, ptr %pointer
  %next = add nuw i64 %i, 2
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %multiple_loop
exit:
  ret void
}
; CHECK-LABEL: IV Users for loop %multiple_loop
; CHECK-NOT: in  %only_assumed =

; The assumed expression itself is observable here and must still be tracked.
define void @observable_operand(ptr %out, ptr %indices, i64 %n, i64 %limit) {
entry:
  br label %operand_loop
operand_loop:
  %i = phi i64 [ 0, %entry ], [ %next, %operand_loop ]
  %second = or disjoint i64 %i, 1
  %assumed_ok = icmp ult i64 %second, %limit
  call void @llvm.assume(i1 %assumed_ok)
  %pointer = getelementptr i64, ptr %out, i64 %second
  store i64 %i, ptr %pointer
  %index_pointer = getelementptr i64, ptr %indices, i64 %i
  store i64 %second, ptr %index_pointer
  %next = add nuw i64 %i, 2
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %operand_loop
exit:
  ret void
}
; CHECK-LABEL: IV Users for loop %operand_loop
; CHECK-NOT: in  %assumed_ok =
; CHECK: in  store i64 %second
; CHECK-NOT: in  %assumed_ok =
