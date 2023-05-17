; RUN: llc -march=hexagon -mcpu=hexagonv5 -disable-hsdr < %s | FileCheck %s

; Check that store is post-incremented.
; CHECK: memh(r{{[0-9]+}}++#2)

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define void @matrix_add_const(i32 %N, ptr nocapture %A, i16 signext %val) #0 {
entry:
  %cmp5 = icmp eq i32 %N, 0
  br i1 %cmp5, label %for.end, label %polly.cond

for.end.loopexit:                                 ; preds = %polly.stmt.for.body29
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %polly.loop_header24.preheader, %entry
  ret void

polly.cond:                                       ; preds = %entry
  %0 = icmp sgt i32 %N, 3
  br i1 %0, label %polly.then, label %polly.loop_header24.preheader

polly.then:                                       ; preds = %polly.cond
  %1 = add i32 %N, -1
  %leftover_lb = and i32 %1, -4
  %2 = icmp sgt i32 %leftover_lb, 0
  br i1 %2, label %polly.loop_body.lr.ph, label %polly.loop_header24.preheader

polly.loop_body.lr.ph:                            ; preds = %polly.then
  %3 = insertelement <4 x i16> undef, i16 %val, i32 0
  %4 = insertelement <4 x i16> %3, i16 %val, i32 1
  %5 = insertelement <4 x i16> %4, i16 %val, i32 2
  %6 = insertelement <4 x i16> %5, i16 %val, i32 3
  br label %polly.loop_body

polly.loop_header24.preheader.loopexit:           ; preds = %polly.loop_body
  br label %polly.loop_header24.preheader

polly.loop_header24.preheader:                    ; preds = %polly.loop_header24.preheader.loopexit, %polly.then, %polly.cond
  %polly.loopiv27.ph = phi i32 [ 0, %polly.cond ], [ %leftover_lb, %polly.then ], [ %leftover_lb, %polly.loop_header24.preheader.loopexit ]
  %7 = icmp slt i32 %polly.loopiv27.ph, %N
  br i1 %7, label %polly.stmt.for.body29.preheader, label %for.end

polly.stmt.for.body29.preheader:                  ; preds = %polly.loop_header24.preheader
  br label %polly.stmt.for.body29

polly.loop_body:                                  ; preds = %polly.loop_body.lr.ph, %polly.loop_body
  %p_arrayidx.phi = phi ptr [ %A, %polly.loop_body.lr.ph ], [ %p_arrayidx.inc, %polly.loop_body ]
  %polly.loopiv34 = phi i32 [ 0, %polly.loop_body.lr.ph ], [ %polly.next_loopiv, %polly.loop_body ]
  %polly.next_loopiv = add nsw i32 %polly.loopiv34, 4
  %_p_vec_full = load <4 x i16>, ptr %p_arrayidx.phi, align 2
  %addp_vec = add <4 x i16> %_p_vec_full, %6
  store <4 x i16> %addp_vec, ptr %p_arrayidx.phi, align 2
  %8 = icmp slt i32 %polly.next_loopiv, %leftover_lb
  %p_arrayidx.inc = getelementptr i16, ptr %p_arrayidx.phi, i32 4
  br i1 %8, label %polly.loop_body, label %polly.loop_header24.preheader.loopexit

polly.stmt.for.body29:                            ; preds = %polly.stmt.for.body29.preheader, %polly.stmt.for.body29
  %polly.loopiv2733 = phi i32 [ %polly.next_loopiv28, %polly.stmt.for.body29 ], [ %polly.loopiv27.ph, %polly.stmt.for.body29.preheader ]
  %polly.next_loopiv28 = add nsw i32 %polly.loopiv2733, 1
  %p_arrayidx30 = getelementptr i16, ptr %A, i32 %polly.loopiv2733
  %_p_scalar_ = load i16, ptr %p_arrayidx30, align 2
  %p_add = add i16 %_p_scalar_, %val
  store i16 %p_add, ptr %p_arrayidx30, align 2
  %exitcond = icmp eq i32 %polly.next_loopiv28, %N
  br i1 %exitcond, label %for.end.loopexit, label %polly.stmt.for.body29
}

attributes #0 = { nounwind "fp-contract-model"="standard" "frame-pointer"="non-leaf" "realign-stack" "relocation-model"="static" "ssp-buffers-size"="8" }
