; RUN: opt -S -analyze -stack-safety-local < %s | FileCheck %s --check-prefixes=CHECK,LOCAL
; RUN: opt -S -passes="print<stack-safety-local>" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,LOCAL
; RUN: opt -S -analyze -stack-safety < %s | FileCheck %s --check-prefixes=CHECK,GLOBAL
; RUN: opt -S -passes="print-stack-safety" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL

; Regression test that exercises a case when a AllocaOffsetRewritten SCEV
; could return an empty-set range. This could occur with udiv SCEVs where the
; RHS was re-written to 0.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @ExternalFn(i64)

define void @Test1() {
; CHECK-LABEL: @Test1 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[1]: empty-set, @Divide1(arg0, full-set){{$}}
; GLOBAL-NEXT: x[1]: full-set, @Divide1(arg0, full-set){{$}}
; CHECK-NOT: ]:
  %x = alloca i8
  %int = ptrtoint i8* %x to i64
  call void @Divide1(i64 %int)
  ret void
}

define dso_local void @Divide1(i64 %arg) {
; CHECK-LABEL: @Divide1{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: arg[]: empty-set, @ExternalFn(arg0, full-set){{$}}
; GLOBAL-NEXT: arg[]: full-set, @ExternalFn(arg0, full-set){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NOT: ]:
  %quotient = udiv i64 undef, %arg
  call void @ExternalFn(i64 %quotient)
  unreachable
}

define void @Test2(i64 %arg) {
; CHECK-LABEL: @Test2 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: arg[]: empty-set{{$}}
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[1]: empty-set, @Divide2(arg0, full-set){{$}}
; GLOBAL-NEXT: x[1]: full-set, @Divide2(arg0, full-set){{$}}
; CHECK-NOT: ]:
  %x = alloca i8
  %int = ptrtoint i8* %x to i64
  call void @Divide2(i64 %int)
  ret void
}

define dso_local void @Divide2(i64 %arg) {
; CHECK-LABEL: @Divide2{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: arg[]: full-set{{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NOT: ]:
  %x = inttoptr i64 %arg to i8*
  %quotient = udiv i64 undef, %arg
  %arrayidx = getelementptr i8, i8* %x, i64 %quotient
  load i8, i8* %arrayidx
  unreachable
}
