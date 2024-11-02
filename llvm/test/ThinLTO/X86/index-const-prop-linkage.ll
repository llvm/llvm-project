; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/index-const-prop-linkage.ll -o %t2.bc
; RUN: llvm-lto2 run -save-temps %t2.bc -r=%t2.bc,foo,pl -r=%t2.bc,g1,pl -r=%t2.bc,g2,pl -r=%t2.bc,g3, \
; RUN:                           %t1.bc -r=%t1.bc,foo, -r=%t1.bc,main,plx -r=%t1.bc,g2,  -o %t3
; RUN: llvm-dis %t3.2.3.import.bc -o - | FileCheck %s

; Check that we never internalize anything with:
; - appending linkage
; - common linkage
; - available_externally linkage
; - reference from @llvm.used
; CHECK:      @llvm.used = appending global [1 x ptr] [ptr @g2]
; CHECK-NEXT: @g1 = external global i32, align 4
; CHECK-NEXT: @g2 = available_externally global i32 42, align 4
; CHECK-NEXT: @g3 = available_externally global i32 42, align 4

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @foo()
@g2 = external global i32
@llvm.used = appending global [1 x i32*] [i32* @g2]

define i32 @main() {
  %v = call i32 @foo()
  ret i32 %v
}
