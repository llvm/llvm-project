; RUN: opt -S -passes="print<stack-safety-local>" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK
; RUN: opt -S -passes="print-stack-safety" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"

@blob = external global [16 x i8]

define dso_local void @CallGlobalVariable(ptr noundef %uaddr) local_unnamed_addr {
; CHECK-LABEL: @CallGlobalVariable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: uaddr[]: full-set{{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  tail call i64 @blob(ptr noundef %uaddr)
  ret void
}

define dso_local void @CallGlobalVariableAlloca() local_unnamed_addr {
; CHECK-LABEL: @CallGlobalVariableAlloca{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[4]: full-set{{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call i64 @blob(ptr noundef %x)
  ret void
}
