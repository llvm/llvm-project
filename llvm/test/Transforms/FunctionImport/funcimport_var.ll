; This test makes sure a static var is not selected as a callee target
; (which will crash compilation).
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_var2.ll -o %t2.bc
; RUN: llvm-lto -thinlto -thinlto-action=thinlink -o %t3 %t.bc %t2.bc
; RUN: llvm-lto -thinlto -thinlto-action=import -thinlto-index=%t3 %t.bc %t2.bc
; RUN: llvm-lto -thinlto -thinlto-action=run %t.bc %t2.bc
; RUN: llvm-nm %t.bc.thinlto.o | FileCheck %s
; RUN: llvm-lto2 run %t.bc %t2.bc -o %t.out \
; RUN:   -r %t.bc,_Z4LinkPKcS0_,plx \
; RUN:   -r %t.bc,link,l \
; RUN:   -r %t2.bc,get_link,plx
; RUN: llvm-nm %t.out.1 | FileCheck %s
; CHECK: U link

; REQUIRES: x86-registered-target

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z4LinkPKcS0_(i8*, i8*) local_unnamed_addr  {
  %3 = tail call i32 @link(i8* %0, i8* %1) #2
  ret i32 %3
}

; Function Attrs: nounwind
declare i32 @link(i8*, i8*) local_unnamed_addr 
