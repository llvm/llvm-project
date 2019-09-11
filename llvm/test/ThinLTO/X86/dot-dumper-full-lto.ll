; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/dot-dumper.ll -o %t2.bc
; RUN: llvm-lto2 run -save-temps %t1.bc %t2.bc -o %t3 \
; RUN:  -r=%t1.bc,main,px \
; RUN:  -r=%t1.bc,A, \
; RUN:  -r=%t2.bc,foo,p \
; RUN:  -r=%t2.bc,bar,p \
; RUN:  -r=%t2.bc,A,p \
; RUN:  -r=%t2.bc,B,p
; RUN: cat %t3.index.dot | FileCheck %s

; CHECK: subgraph cluster_4294967295
; CHECK:   M4294967295_[[ID:[0-9]+]]{{.*}}main
; CHECK: // Cross-module edges:
; CHECK:  M4294967295_[[ID]] -> M0_{{[0-9]+}}{{.*}}// ref

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = external global i32

define i32 @main() {
  %v = load i32, i32* @A
  ret i32 %v
}

!0 = !{i32 1, !"ThinLTO", i32 0}
!llvm.module.flags = !{ !0 }
