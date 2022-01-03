; RUN: split-file %s %t
; RUN: opt -module-summary %t/a.ll -o %ta.bc
; RUN: opt -module-summary %t/b.ll -o %tb.bc
; RUN: llvm-lto2 run %ta.bc %tb.bc -o %tc.bc -save-temps \
; RUN:   -r=%ta.bc,nossp_caller,px \
; RUN:   -r=%ta.bc,ssp_caller,px \
; RUN:   -r=%ta.bc,nossp_caller2,px \
; RUN:   -r=%ta.bc,ssp_caller2,px \
; RUN:   -r=%ta.bc,nossp_callee,x \
; RUN:   -r=%ta.bc,ssp_callee,x \
; RUN:   -r=%tb.bc,nossp_callee,px \
; RUN:   -r=%tb.bc,ssp_callee,px \
; RUN:   -r=%tb.bc,foo
; RUN: llvm-dis %tc.bc.1.4.opt.bc -o - | FileCheck %s

;--- a.ll

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare void @nossp_callee()
declare void @ssp_callee() ssp

; nossp caller should be able to inline nossp callee.
define void @nossp_caller() {
; CHECK-LABEL: define void @nossp_caller()
; CHECK-NOT: #0
; CHECK-NEXT: tail call void @foo
  tail call void @nossp_callee()
  ret void
}

; ssp caller should be able to inline ssp callee.
define void @ssp_caller() ssp {
; CHECK-LABEL: define void @ssp_caller()
; CHECK-SAME: #0
; CHECK-NEXT: tail call void @foo
  tail call void @ssp_callee()
  ret void
}

; nossp caller should be able to inline ssp callee.
; the ssp attribute is not propagated.
define void @nossp_caller2() {
; CHECK-LABEL: define void @nossp_caller2()
; CHECK-NOT: #0
; CHECK-NEXT: tail call void @foo
  tail call void @ssp_callee()
  ret void
}

; ssp caller should be able to inline nossp callee.
define void @ssp_caller2() ssp {
; CHECK-LABEL: define void @ssp_caller2()
; CHECK-SAME: #0
; CHECK-NEXT: tail call void @foo
  tail call void @nossp_callee()
  ret void
}

; CHECK: attributes #0 = { ssp }

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare void @foo()

define void @nossp_callee() {
  call void @foo()
  ret void
}

define void @ssp_callee() ssp {
  call void @foo()
  ret void
}
