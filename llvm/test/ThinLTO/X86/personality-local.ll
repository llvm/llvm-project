; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %S/Inputs/personality-local.ll -o %t2.bc

; RUN: llvm-lto2 run -o %t.o %t1.bc %t2.bc -save-temps \
; RUN:   -r %t2.bc,foo,p \
; RUN:   -r %t1.bc,foo,l \
; RUN:   -r %t1.bc,bar,p \
; RUN:   -r %t1.bc,main,xp
; RUN: llvm-readobj --symbols %t.o.2 | FileCheck %s

; CHECK:      Symbol {
; CHECK:        Name: foo
; CHECK-NEXT:   Value: 0x0
; CHECK-NEXT:   Size: 1
; CHECK-NEXT:   Binding: Global
; CHECK-NEXT:   Type: Function
; CHECK-NEXT:   Other: 0
; CHECK-NEXT:   Section: .text
; CHECK-NEXT: }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare void @foo()

define void @bar() personality ptr @personality_routine {
 ret void
}

define internal i32 @personality_routine(i32, i32, i64, ptr, ptr) {
  call void @foo()
  ret i32 0
}

define i32 @main() {
  call void @bar()
  ret i32 0
}

