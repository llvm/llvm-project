; RUN: opt %loadPolly -polly-detect -disable-output < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

declare i32 @generic_personality_v0(i32, i64, i8*, i8*)

define i16 @v() personality i32 (i32, i64, i8*, i8*)* @generic_personality_v0 {
entry:
  br i1 undef, label %bb16, label %invcont12

invcont12:                                        ; preds = %invcont11
  %a = invoke i16 @v() to label %return unwind label %lpad22   ; <i16*> [#uses=1]

bb16:                                             ; preds = %bb7
  br i1 undef, label %bb9, label %return

return:                                           ; preds = %bb16, %invcont12
  %b = phi i16 [ %a, %invcont12 ], [ 0, %bb16 ] ; <i16*> [#uses=1]
  ret i16 %b

bb9:                                             ; preds = %bb3
  ret i16 0

lpad22:                                           ; preds = %invcont12
  %d = landingpad { i8*, i32 } cleanup catch i32* null
  unreachable
}
