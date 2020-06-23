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

define i64 addrspace(4)* @memset(i1 %alwaysFalse) {
; CHECK-LABEL: @memset(
; CHECK-NOT: inttoptr
; CHECK-NOT: ptrtoint
entry:
  %x = alloca i64 addrspace(4)*
  %cast.0 = bitcast i64 addrspace(4)** %x to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %cast.0, i8 5, i64 16, i1 false)
  br i1 %alwaysFalse, label %neverTaken, label %alwaysTaken

neverTaken:
  %x.field.ld.0 = load i64 addrspace(4)*, i64 addrspace(4)** %x
  ret i64 addrspace(4)* %x.field.ld.0
  
alwaysTaken:
  ret i64 addrspace(4)* null
}

;; TODO: This one demonstrates a missed oppurtunity.  The only known bit
;; pattern for a non-integral bit pattern is that null is zero.  As such
;; we could do SROA and replace the memset w/a null store.  This will
;; usually be gotten by instcombine.
define i64 addrspace(4)* @memset_null(i1 %alwaysFalse) {
; CHECK-LABEL: @memset_null(
; CHECK-NOT: inttoptr
; CHECK-NOT: ptrtoint
entry:
  %x = alloca i64 addrspace(4)*
  %cast.0 = bitcast i64 addrspace(4)** %x to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %cast.0, i8 0, i64 16, i1 false)
  br i1 %alwaysFalse, label %neverTaken, label %alwaysTaken

neverTaken:
  %x.field.ld.0 = load i64 addrspace(4)*, i64 addrspace(4)** %x
  ret i64 addrspace(4)* %x.field.ld.0
  
alwaysTaken:
  ret i64 addrspace(4)* null
}

%union.anon = type { i32* }

; CHECK-LABEL: @f2(
; CHECK-NOT: ptr2int
; CHECK-NOT: int2ptr
define i8 *@f2(i8 addrspace(4)* %p) {
  %1 = alloca %union.anon, align 8
  %2 = bitcast %union.anon* %1 to i8 addrspace(4)**
  store i8 addrspace(4)* %p, i8 addrspace(4)** %2, align 8
  %3 = bitcast %union.anon* %1 to i8**
  %4 = load i8*, i8** %3, align 8
  ret i8* %4
}

declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1)
