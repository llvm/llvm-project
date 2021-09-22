; RUN: opt %loadPolly -polly-codegen -disable-output < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
define void @cfft2([2 x float]* %x) nounwind {
entry:
  %d.1.reg2mem = alloca [2 x float]*              ; <[2 x float]**> [#uses=3]
  br i1 undef, label %bb2, label %bb34

bb2:                                              ; preds = %bb34, %entry
  ret void

bb20:                                             ; preds = %bb34
  store [2 x float]* undef, [2 x float]** %d.1.reg2mem
  br i1 false, label %bb21, label %bb23

bb21:                                             ; preds = %bb20
  %0 = getelementptr inbounds [2 x float], [2 x float]* %x, i64 undef ; <[2 x float]*> [#uses=1]
  store [2 x float]* %0, [2 x float]** %d.1.reg2mem
  br label %bb23

bb23:                                             ; preds = %bb21, %bb20
  %d.1.reload = load [2 x float]*, [2 x float]** %d.1.reg2mem   ; <[2 x float]*> [#uses=1]
  br i1 undef, label %bb29, label %bb34

bb29:                                             ; preds = %bb23
  %1 = getelementptr inbounds [2 x float], [2 x float]* %d.1.reload, i64 undef ; <[2 x float]*> [#uses=0]
  br label %bb34

bb34:                                             ; preds = %bb29, %bb23, %entry
  br i1 undef, label %bb20, label %bb2
}
