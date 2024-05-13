; PR2463
; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/2008-07-06-AliasWeakDest2.ll -o %t2.bc
; RUN: llvm-link %t1.bc %t2.bc -o %t3.bc
; RUN: llvm-link %t2.bc %t1.bc -o %t4.bc

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

@sched_clock = alias i64 (), ptr @native_sched_clock

@foo = alias i32, ptr @realfoo
@realfoo = global i32 0

define i64 @native_sched_clock() nounwind  {
entry:
        ret i64 0
}
