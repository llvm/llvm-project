; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < %p/2008-03-07-DroppedSection_a.ll > %t2.bc
; RUN: llvm-ld -r -disable-opt %t.bc %t2.bc -o %t3.bc
; RUN: llvm-dis < %t3.bc | grep ".data.init_task"

; ModuleID = 'u.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"
@init_task_union = external global i32

