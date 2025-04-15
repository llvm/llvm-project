; RUN: opt -passes=slp-vectorizer -S %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64-S128"
target triple = "x86_64-apple-macosx"

; This test used to fail under libgmalloc. Because we would try to access a
; pointer that was already deleted.
;
; llvm-lit -v --param use_gmalloc=1 --param
;   gmalloc_path=/usr/lib/libgmalloc.dylib
;   test/Transforms/LoopVectorize/X86/rauw-bug.ll
;
; radar://15498655

; CHECK: reduced
define void @reduced(i1 %arg)  {
entry:
  br i1 %arg, label %while.body, label %while.cond63.preheader.while.end76_crit_edge

while.cond63.preheader.while.end76_crit_edge:
  ret void

while.body:
  %d2_fx.015 = phi double [ %sub52, %while.body ], [ undef, %entry ]
  %d2_fy.014 = phi double [ %sub58, %while.body ], [ undef, %entry ]
  %d3_fy.013 = phi double [ %div56, %while.body ], [ undef, %entry ]
  %d3_fx.012 = phi double [ %div50, %while.body ], [ undef, %entry ]
  %div50 = fmul double %d3_fx.012, 1.250000e-01
  %sub52 = fsub double 0.000000e+00, %div50
  %div56 = fmul double %d3_fy.013, 1.250000e-01
  %sub58 = fsub double 0.000000e+00, %div56
  br label %while.body
}
