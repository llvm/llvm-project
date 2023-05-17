; RUN: opt -passes=newgvn -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:4"
target triple = "x86_64-unknown-linux-gnu"

define void @f0(i1 %alwaysFalse, i64 %val, ptr %loc) {
; CHECK-LABEL: @f0(
; CHECK-NOT: inttoptr
; CHECK-NOT: ptrtoint
 entry:
  store i64 %val, ptr %loc
  br i1 %alwaysFalse, label %neverTaken, label %alwaysTaken

 neverTaken:
  %ptr = load ptr addrspace(4), ptr %loc
  store i8 5, ptr addrspace(4) %ptr
  ret void

 alwaysTaken:
  ret void
}

define i64 @f1(i1 %alwaysFalse, ptr addrspace(4) %val, ptr %loc) {
; CHECK-LABEL: @f1(
; CHECK-NOT: inttoptr
; CHECK-NOT: ptrtoint
 entry:
  store ptr addrspace(4) %val, ptr %loc
  br i1 %alwaysFalse, label %neverTaken, label %alwaysTaken

 neverTaken:
  %int = load i64, ptr %loc
  ret i64 %int

 alwaysTaken:
  ret i64 42
}
