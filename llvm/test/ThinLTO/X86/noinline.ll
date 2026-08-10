; This test checks that ThinLTO doesn't try to import noinline function
; which, when takes place, causes promotion of its callee.
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/noinline.ll -o %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t3.o \
; RUN:   -save-temps       \
; RUN:   -r=%t1.bc,main,px \
; RUN:   -r=%t1.bc,foo,    \
; RUN:   -r=%t2.bc,foo,p

; RUN: llvm-dis %t3.o.1.3.import.bc -o - | FileCheck %s

; CHECK-NOT: define available_externally i32 @foo

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32, ptr nocapture readnone) local_unnamed_addr #0 {
  %3 = tail call i32 @foo(i32 %0) #0
  ret i32 %3
}

declare i32 @foo(i32) local_unnamed_addr

attributes #0 = { nounwind }
