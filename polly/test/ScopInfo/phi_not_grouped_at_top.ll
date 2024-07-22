; RUN: opt %loadNPMPolly -passes=polly-prepare -disable-output < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

declare i32 @funa() align 2

declare i32 @generic_personality_v0(i32, i64, ptr, ptr)

define void @funb() personality ptr @generic_personality_v0 {
entry:
  br label %bb117

bb117:                                            ; preds = %bb56
  %0 = invoke i32 @funa()
          to label %bb121 unwind label %invcont118 ; <ptr> [#uses=1]

invcont118:                                       ; preds = %bb117
  %d = landingpad { ptr, i32 } cleanup catch ptr null
  br label %bb121

bb121:                                            ; preds = %bb120, %invcont118
  %iftmp.82.0 = phi i32 [ 0, %bb117 ], [ 1, %invcont118 ] ; <i8> [#uses=1]
  %te.1 = phi i32 [ undef, %invcont118 ], [ %0, %bb117 ] ;
  %cnd = icmp ne i32 %iftmp.82.0, %te.1          ; <i1> [#uses=1]
  br label %return

return:                                           ; preds = %entry
  ret void
}
