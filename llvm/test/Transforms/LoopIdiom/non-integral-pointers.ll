; RUN: opt -S -basicaa -loop-idiom < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:4"
target triple = "x86_64-unknown-linux-gnu"

define void @f_0(i8 addrspace(3)** %ptr) {
; CHECK-LABEL: @f_0(
; CHECK: call{{.*}}memset

; LIR'ing stores of pointers with address space 3 is fine, since
; they're integral pointers.

entry:
  br label %for.body

for.body:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr i8 addrspace(3)*, i8 addrspace(3)** %ptr, i64 %indvar
  store i8 addrspace(3)* null, i8 addrspace(3)** %arrayidx, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @f_1(i8 addrspace(4)** %ptr) {
; CHECK-LABEL: @f_1(
; CHECK-NOT: call{{.*}}memset

; LIR'ing stores of pointers with address space 4 is not ok, since
; they're non-integral pointers.

entry:
  br label %for.body

for.body:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr i8 addrspace(4)*, i8 addrspace(4)** %ptr, i64 %indvar
  store i8 addrspace(4)* null, i8 addrspace(4)** %arrayidx, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
