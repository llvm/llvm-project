; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/globals-import-blockaddr.ll -o %t2.bc
; RUN: llvm-lto2 run -save-temps %t1.bc -r=%t1.bc,foo,l -r=%t1.bc,main,pl %t2.bc -r=%t2.bc,foo,pl -o %t3
; RUN: llvm-dis %t3.1.3.import.bc -o - | FileCheck %s

; Verify that we haven't imported GV containing blockaddress
; CHECK: @label_addr.llvm.0 = external hidden constant

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local [1 x i8*]* @foo();

define dso_local i32 @main() {
  %p = call [1 x i8*]* @foo()
  %v = ptrtoint [1 x i8*]* %p to i32
  ret i32 %v
}
