; RUN: llvm-as %s -o %t0.o
; RUN: llvm-as < %p/Inputs/largedatathreshold.ll > %t1.o
; RUN: llvm-lto2 run -save-temps -r %t0.o,_start,px -r %t1.o,bar,px -r %t1.o,large_data,px -r %t1.o,small_data,px \
; RUN:   -r %t0.o,_GLOBAL_OFFSET_TABLE_, -r %t1.o,_GLOBAL_OFFSET_TABLE_, %t0.o %t1.o -o %t2.s
; RUN: llvm-dis %t2.s.0.0.preopt.bc -o - | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@data = internal constant [20 x i8] zeroinitializer

define ptr @_start() {
entry:
  ret ptr @data
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 9, !"Code Model", i32 3}
!1 = !{i32 9, !"Large Data Threshold", i32 100}

; CHECK: @data = internal constant [20 x i8]
; CHECK: @large_data = hidden global [105 x i8] {{.*}}, code_model "large"
; CHECK: @small_data = hidden global [50 x i8] {{.*}}, code_model "small"
; CHECK: define ptr @_start()
; CHECK: define void @bar()
