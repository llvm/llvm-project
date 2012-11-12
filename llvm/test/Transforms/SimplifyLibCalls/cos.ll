; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define double @foo(double %d) nounwind readnone {
; CHECK: @foo
    %1 = fsub double -0.000000e+00, %d
    %2 = call double @cos(double %1) nounwind readnone
; CHECK: call double @cos(double %d)
    ret double %2
}

declare double @cos(double) nounwind readnone
