; RUN: llc < %s -O3 -relocation-model=pic -frame-pointer=all -mcpu=nocona
;
; This test case is reduced from Bullet. It crashes SSEDomainFix.
;
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0"

declare i32 @_ZN11HullLibrary16CreateConvexHullERK8HullDescR10HullResult(ptr, ptr nocapture, ptr nocapture) ssp align 2

define void @_ZN17btSoftBodyHelpers4DrawEP10btSoftBodyP12btIDebugDrawi(ptr %psb, ptr %idraw, i32 %drawflags) ssp align 2 personality ptr @__gxx_personality_v0 {
entry:
  br i1 undef, label %bb92, label %bb58

bb58:                                             ; preds = %entry
  %0 = invoke i32 @_ZN11HullLibrary16CreateConvexHullERK8HullDescR10HullResult(ptr undef, ptr undef, ptr undef)
          to label %invcont64 unwind label %lpad159 ; <i32> [#uses=0]

invcont64:                                        ; preds = %bb58
  br i1 undef, label %invcont65, label %bb.i.i

bb.i.i:                                           ; preds = %invcont64
  %1 = load <4 x float>, ptr undef, align 16          ; <<4 x float>> [#uses=5]
  br i1 undef, label %bb.nph.i.i, label %invcont65

bb.nph.i.i:                                       ; preds = %bb.i.i
  %tmp22.i.i = bitcast <4 x float> %1 to i128     ; <i128> [#uses=1]
  %tmp23.i.i = trunc i128 %tmp22.i.i to i32       ; <i32> [#uses=1]
  %2 = bitcast i32 %tmp23.i.i to float            ; <float> [#uses=1]
  %tmp6.i = extractelement <4 x float> %1, i32 1  ; <float> [#uses=1]
  %tmp2.i = extractelement <4 x float> %1, i32 2  ; <float> [#uses=1]
  br label %bb1.i.i

bb1.i.i:                                          ; preds = %bb1.i.i, %bb.nph.i.i
  %.tmp6.0.i.i = phi float [ %tmp2.i, %bb.nph.i.i ], [ %5, %bb1.i.i ] ; <float> [#uses=1]
  %.tmp5.0.i.i = phi float [ %tmp6.i, %bb.nph.i.i ], [ %4, %bb1.i.i ] ; <float> [#uses=1]
  %.tmp.0.i.i = phi float [ %2, %bb.nph.i.i ], [ %3, %bb1.i.i ] ; <float> [#uses=1]
  %3 = fadd float %.tmp.0.i.i, undef              ; <float> [#uses=2]
  %4 = fadd float %.tmp5.0.i.i, undef             ; <float> [#uses=2]
  %5 = fadd float %.tmp6.0.i.i, undef             ; <float> [#uses=2]
  br i1 undef, label %bb2.return.loopexit_crit_edge.i.i, label %bb1.i.i

bb2.return.loopexit_crit_edge.i.i:                ; preds = %bb1.i.i
  %tmp8.i = insertelement <4 x float> %1, float %3, i32 0 ; <<4 x float>> [#uses=1]
  %tmp4.i = insertelement <4 x float> %tmp8.i, float %4, i32 1 ; <<4 x float>> [#uses=1]
  %tmp.i = insertelement <4 x float> %tmp4.i, float %5, i32 2 ; <<4 x float>> [#uses=1]
  br label %invcont65

invcont65:                                        ; preds = %bb2.return.loopexit_crit_edge.i.i, %bb.i.i, %invcont64
  %.0.i = phi <4 x float> [ %tmp.i, %bb2.return.loopexit_crit_edge.i.i ], [ undef, %invcont64 ], [ %1, %bb.i.i ] ; <<4 x float>> [#uses=1]
  %tmp15.i = extractelement <4 x float> %.0.i, i32 2 ; <float> [#uses=1]
  %6 = fmul float %tmp15.i, undef                 ; <float> [#uses=1]
  br label %bb.i265

bb.i265:                                          ; preds = %bb.i265, %invcont65
  %7 = fsub float 0.000000e+00, %6                ; <float> [#uses=1]
  store float %7, ptr undef, align 4
  br label %bb.i265

bb92:                                             ; preds = %entry
  unreachable

lpad159:                                          ; preds = %bb58
  %exn = landingpad {ptr, i32}
            cleanup
  unreachable
}

declare i32 @__gxx_personality_v0(...)
