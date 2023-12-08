; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s

; Check that we do not generate extract.
; CHECK-NOT: extractu
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define void @foo(i32 %N, ptr nocapture %C, ptr nocapture %A, i16 signext %val) #0 {
entry:
  %cmp14 = icmp eq i32 %N, 0
  br i1 %cmp14, label %for.end11, label %for.cond1.preheader.single_entry.preheader

for.cond1.preheader.single_entry.preheader:       ; preds = %entry
  %0 = add i32 %N, -1
  %leftover_lb = and i32 %0, -2
  %p_conv4 = sext i16 %val to i32
  br label %for.cond1.preheader.single_entry

for.cond1.preheader.single_entry:                 ; preds = %for.inc9, %for.cond1.preheader.single_entry.preheader
  %indvar = phi i32 [ %indvar.next, %for.inc9 ], [ 0, %for.cond1.preheader.single_entry.preheader ]
  %1 = mul i32 %indvar, %N
  %.not = icmp slt i32 %N, 2
  %.not41 = icmp slt i32 %leftover_lb, 1
  %brmerge = or i1 %.not, %.not41
  %.mux = select i1 %.not, i32 0, i32 %leftover_lb
  br i1 %brmerge, label %polly.loop_header26.preheader, label %polly.loop_body.lr.ph

for.inc9.loopexit:                                ; preds = %polly.stmt.for.body331
  br label %for.inc9

for.inc9:                                         ; preds = %for.inc9.loopexit, %polly.loop_header26.preheader
  %indvar.next = add i32 %indvar, 1
  %exitcond40 = icmp eq i32 %indvar.next, %N
  br i1 %exitcond40, label %for.end11.loopexit, label %for.cond1.preheader.single_entry

for.end11.loopexit:                               ; preds = %for.inc9
  br label %for.end11

for.end11:                                        ; preds = %for.end11.loopexit, %entry
  ret void

polly.loop_body.lr.ph:                            ; preds = %for.cond1.preheader.single_entry
  %2 = call i64 @llvm.hexagon.A2.combinew(i32 %1, i32 %1)
  %3 = bitcast i64 %2 to <2 x i32>
  %4 = extractelement <2 x i32> %3, i32 0
  %5 = call i64 @llvm.hexagon.A2.combinew(i32 %p_conv4, i32 %p_conv4)
  %6 = bitcast i64 %5 to <2 x i32>
  %p_arrayidx8.gep = getelementptr i32, ptr %C, i32 %4
  %p_arrayidx.gep = getelementptr i16, ptr %A, i32 %4
  br label %polly.loop_body

polly.loop_body:                                  ; preds = %polly.loop_body.lr.ph, %polly.loop_body
  %p_arrayidx8.phi = phi ptr [ %p_arrayidx8.gep, %polly.loop_body.lr.ph ], [ %p_arrayidx8.inc, %polly.loop_body ]
  %p_arrayidx.phi = phi ptr [ %p_arrayidx.gep, %polly.loop_body.lr.ph ], [ %p_arrayidx.inc, %polly.loop_body ]
  %polly.loopiv38 = phi i32 [ 0, %polly.loop_body.lr.ph ], [ %polly.next_loopiv, %polly.loop_body ]
  %polly.next_loopiv = add nsw i32 %polly.loopiv38, 2
  %_p_vec_full = load <2 x i16>, ptr %p_arrayidx.phi, align 2
  %7 = sext <2 x i16> %_p_vec_full to <2 x i32>
  %mul5p_vec = mul <2 x i32> %7, %6
  store <2 x i32> %mul5p_vec, ptr %p_arrayidx8.phi, align 4
  %8 = icmp slt i32 %polly.next_loopiv, %leftover_lb
  %p_arrayidx8.inc = getelementptr i32, ptr %p_arrayidx8.phi, i32 2
  %p_arrayidx.inc = getelementptr i16, ptr %p_arrayidx.phi, i32 2
  br i1 %8, label %polly.loop_body, label %polly.loop_header26.preheader.loopexit

polly.loop_header26.preheader.loopexit:           ; preds = %polly.loop_body
  br label %polly.loop_header26.preheader

polly.loop_header26.preheader:                    ; preds = %polly.loop_header26.preheader.loopexit, %for.cond1.preheader.single_entry
  %polly.loopiv29.ph = phi i32 [ %.mux, %for.cond1.preheader.single_entry ], [ %leftover_lb, %polly.loop_header26.preheader.loopexit ]
  %9 = icmp slt i32 %polly.loopiv29.ph, %N
  br i1 %9, label %polly.stmt.for.body331.preheader, label %for.inc9

polly.stmt.for.body331.preheader:                 ; preds = %polly.loop_header26.preheader
  br label %polly.stmt.for.body331

polly.stmt.for.body331:                           ; preds = %polly.stmt.for.body331.preheader, %polly.stmt.for.body331
  %polly.loopiv2939 = phi i32 [ %polly.next_loopiv30, %polly.stmt.for.body331 ], [ %polly.loopiv29.ph, %polly.stmt.for.body331.preheader ]
  %polly.next_loopiv30 = add nsw i32 %polly.loopiv2939, 1
  %p_32 = add i32 %polly.loopiv2939, %1
  %p_arrayidx833 = getelementptr i32, ptr %C, i32 %p_32
  %p_arrayidx34 = getelementptr i16, ptr %A, i32 %p_32
  %_p_scalar_ = load i16, ptr %p_arrayidx34, align 2
  %p_conv = sext i16 %_p_scalar_ to i32
  %p_mul5 = mul nsw i32 %p_conv, %p_conv4
  store i32 %p_mul5, ptr %p_arrayidx833, align 4
  %exitcond = icmp eq i32 %polly.next_loopiv30, %N
  br i1 %exitcond, label %for.inc9.loopexit, label %polly.stmt.for.body331
}

declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
