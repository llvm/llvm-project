; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/distributed_import.ll -o %t2.bc

; RUN: llvm-lto2 %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -thinlto-distributed-indexes \
; RUN:     -r=%t1.bc,g, \
; RUN:     -r=%t1.bc,f,px \
; RUN:     -r=%t2.bc,g,px
; RUN:  opt -function-import -summary-file %t1.bc.thinlto.bc %t1.bc -o %t1.out
; RUN: opt -function-import -summary-file %t2.bc.thinlto.bc %t2.bc -o %t2.out
; RUN: llvm-dis -o - %t2.out | FileCheck %s
; CHECK: @G.llvm.0

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i32 @g(...)

define void @f() {
entry:
  call i32 (...) @g()
  ret void
}
