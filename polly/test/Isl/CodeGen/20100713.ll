; RUN: opt %loadPolly %defaultOpts -polly-codegen-isl %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @fft_float(i32 %NumSamples) nounwind {
  br label %bb18

bb18:                                             ; preds = %bb17
  br i1 false, label %bb19, label %bb22

bb19:                                             ; preds = %bb18
  %a = uitofp i32 %NumSamples to double           ; <double> [#uses=1]
  br label %bb21

bb20:                                             ; preds = %bb21
  %1 = load float* undef, align 4                 ; <float> [#uses=0]
  %2 = fpext float undef to double                ; <double> [#uses=1]
  %3 = fdiv double %2, %a ; <double> [#uses=0]
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %bb21

bb21:                                             ; preds = %bb20, %bb19
  %indvar = phi i64 [ %indvar.next, %bb20 ], [ 0, %bb19 ] ; <i64> [#uses=1]
  br i1 false, label %bb20, label %bb22.loopexit

bb22.loopexit:                                    ; preds = %bb21
  br label %bb22

bb22:                                             ; preds = %bb22.loopexit, %bb18
  br label %return

return:                                           ; preds = %bb22
  ret void
}
