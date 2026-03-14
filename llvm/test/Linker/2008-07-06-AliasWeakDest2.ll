; This file is used by 2008-07-06-AliasWeakDest2.ll
; RUN: true

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

@foo = weak global i32 2

define i64 @sched_clock_cpu(i32 inreg  %cpu) nounwind  {
entry:
        %tmp = call i64 @sched_clock( ) nounwind                ; <i64>
        ret i64 %tmp
}

define weak i64 @sched_clock() {
entry:
        ret i64 1
}
