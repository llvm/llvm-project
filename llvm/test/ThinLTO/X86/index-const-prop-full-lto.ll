; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/index-const-prop-define-g.ll -o %t2.bc
; RUN: opt -module-summary %p/Inputs/index-const-prop-full-lto.ll -o %t3.bc
; RUN: llvm-lto2 run -save-temps %t2.bc -r=%t2.bc,g,pl \
; RUN:                 %t1.bc -r=%t1.bc,foo,l -r=%t1.bc,main,plx -r=%t1.bc,g, \
; RUN:                 %t3.bc -r=%t3.bc,foo,pl -r=%t3.bc,g, -o %t4
; RUN: llvm-dis %t4.2.3.import.bc -o - | FileCheck %s

; All references from functions in full LTO module are not constant.
; We cannot internalize @g
; CHECK: @g = available_externally global i32 42

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @foo()
@g = external global i32

define i32 @main() {
  %v = call i32 @foo()
  %v2 = load i32, ptr @g
  %v3 = add i32 %v, %v2
  ret i32 %v3
}
