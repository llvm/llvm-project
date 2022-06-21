; RUN: opt -passes=sroa -S < %s | FileCheck %s

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
  store i64 %val, ptr %loc
  br i1 %alwaysFalse, label %neverTaken, label %alwaysTaken

neverTaken:
  %ptr = load ptr addrspace(4), ptr %loc
  store i8 5, ptr addrspace(4) %ptr
  ret void

alwaysTaken:
  ret void
}

define i64 @f1(i1 %alwaysFalse, ptr addrspace(4) %val) {
; CHECK-LABEL: @f1(
; CHECK-NOT: inttoptr
; CHECK-NOT: ptrtoint
entry:
  %loc = alloca ptr addrspace(4)
  store ptr addrspace(4) %val, ptr %loc
  br i1 %alwaysFalse, label %neverTaken, label %alwaysTaken

neverTaken:
  %int = load i64, ptr %loc
  ret i64 %int

alwaysTaken:
  ret i64 42
}

define ptr addrspace(4) @memset(i1 %alwaysFalse) {
; CHECK-LABEL: @memset(
; CHECK-NOT: inttoptr
; CHECK-NOT: ptrtoint
entry:
  %x = alloca ptr addrspace(4)
  call void @llvm.memset.p0.i64(ptr align 8 %x, i8 5, i64 16, i1 false)
  br i1 %alwaysFalse, label %neverTaken, label %alwaysTaken

neverTaken:
  %x.field.ld.0 = load ptr addrspace(4), ptr %x
  ret ptr addrspace(4) %x.field.ld.0
  
alwaysTaken:
  ret ptr addrspace(4) null
}

;; TODO: This one demonstrates a missed oppurtunity.  The only known bit
;; pattern for a non-integral bit pattern is that null is zero.  As such
;; we could do SROA and replace the memset w/a null store.  This will
;; usually be gotten by instcombine.
define ptr addrspace(4) @memset_null(i1 %alwaysFalse) {
; CHECK-LABEL: @memset_null(
; CHECK-NOT: inttoptr
; CHECK-NOT: ptrtoint
entry:
  %x = alloca ptr addrspace(4)
  call void @llvm.memset.p0.i64(ptr align 8 %x, i8 0, i64 16, i1 false)
  br i1 %alwaysFalse, label %neverTaken, label %alwaysTaken

neverTaken:
  %x.field.ld.0 = load ptr addrspace(4), ptr %x
  ret ptr addrspace(4) %x.field.ld.0
  
alwaysTaken:
  ret ptr addrspace(4) null
}

%union.anon = type { ptr }

; CHECK-LABEL: @f2(
; CHECK-NOT: ptr2int
; CHECK-NOT: int2ptr
define ptr@f2(ptr addrspace(4) %p) {
  %1 = alloca %union.anon, align 8
  %2 = bitcast ptr %1 to ptr
  store ptr addrspace(4) %p, ptr %2, align 8
  %3 = bitcast ptr %1 to ptr
  %4 = load ptr, ptr %3, align 8
  ret ptr %4
}

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
