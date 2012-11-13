; RUN: opt %loadPolly %defaultOpts -polly-detect < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define fastcc void @f () inlinehint align 2 {
entry:
  %0 = fmul double undef, 1.250000e+00            ; <double> [#uses=1]
  %1 = fptoui double %0 to i32                    ; <i32> [#uses=0]
  br i1 false, label %bb5.i, label %bb.nph.i

bb.nph.i:                                         ; preds = %bb.i1
  br label %bb3.i2

bb3.i2:                                           ; preds = %bb3.i2, %bb.nph.i
  br i1 undef, label %bb3.i2, label %bb5.i

bb5.i:                                            ; preds = %bb3.i2, %bb.i1
  br label %exit

exit:
  ret void
}
