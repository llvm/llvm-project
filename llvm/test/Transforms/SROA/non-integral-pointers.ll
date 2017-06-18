; RUN: opt -sroa -S < %s | FileCheck %s

; This test checks that SROA does not introduce ptrtoint and inttoptr
; casts from and to non-integral pointers.  The "ni:4" bit in the
; datalayout states that pointers of address space 4 are to be
; considered "non-integral".

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:4"
target triple = "x86_64-unknown-linux-gnu"

define void @f0(i1 %alwaysFalse, i64 %val) {
; CHECK-LABEL: @f0(
; CHECK-NOT: inttoptr
; CHECK-NOT: ptrtoint
entry:
  %loc = alloca i64
  store i64 %val, i64* %loc
  br i1 %alwaysFalse, label %neverTaken, label %alwaysTaken

neverTaken:
  %loc.bc = bitcast i64* %loc to i8 addrspace(4)**
  %ptr = load i8 addrspace(4)*, i8 addrspace(4)** %loc.bc
  store i8 5, i8 addrspace(4)* %ptr
  ret void

alwaysTaken:
  ret void
}

define i64 @f1(i1 %alwaysFalse, i8 addrspace(4)* %val) {
; CHECK-LABEL: @f1(
; CHECK-NOT: inttoptr
; CHECK-NOT: ptrtoint
entry:
  %loc = alloca i8 addrspace(4)*
  store i8 addrspace(4)* %val, i8 addrspace(4)** %loc
  br i1 %alwaysFalse, label %neverTaken, label %alwaysTaken

neverTaken:
  %loc.bc = bitcast i8 addrspace(4)** %loc to i64*
  %int = load i64, i64* %loc.bc
  ret i64 %int

alwaysTaken:
  ret i64 42
}
