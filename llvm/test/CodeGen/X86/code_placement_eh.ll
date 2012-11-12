; RUN: llc < %s

; CodePlacementOpt shouldn't try to modify this loop because
; it involves EH edges.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0"

define void @foo() {
invcont5:
  br label %bb15

.noexc3:                                          ; preds = %bb15
  br i1 undef, label %bb18.i5.i, label %bb15

.noexc6.i.i:                                      ; preds = %bb18.i5.i
  %tmp2021 = invoke float @cosf(float 0.000000e+00) readonly
          to label %bb18.i5.i unwind label %lpad.i.i ; <float> [#uses=0]

bb18.i5.i:                                        ; preds = %.noexc6.i.i, %bb51.i
  %tmp2019 = invoke float @sinf(float 0.000000e+00) readonly
          to label %.noexc6.i.i unwind label %lpad.i.i ; <float> [#uses=0]

lpad.i.i:                                         ; preds = %bb18.i5.i, %.noexc6.i.i
  %lpadval.i.i = landingpad { i8*, i32 } personality i32 (...)* @__gxx_personality_v0
          catch i8* null
  unreachable

lpad59.i:                                         ; preds = %bb15
  %lpadval60.i.i = landingpad { i8*, i32 } personality i32 (...)* @__gxx_personality_v0
          catch i8* null
  unreachable

bb15:                                             ; preds = %.noexc3, %invcont5
  invoke fastcc void @_ZN28btHashedOverlappingPairCacheC2Ev()
          to label %.noexc3 unwind label %lpad59.i
}

declare i32 @__gxx_personality_v0(...)

declare float @sinf(float) readonly

declare float @cosf(float) readonly

declare fastcc void @_ZN28btHashedOverlappingPairCacheC2Ev() align 2
