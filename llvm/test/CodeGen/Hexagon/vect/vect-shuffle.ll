; RUN: llc -march=hexagon -mcpu=hexagonv5 -disable-hsdr < %s | FileCheck %s

; Check that store is post-incremented.
; CHECK-NOT: extractu(r{{[0-9]+}},#32,
; CHECK-NOT: insert
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define i32 @foo(i16* noalias nocapture %src, i16* noalias nocapture %dstImg, i32 %width, i32 %idx, i32 %flush) #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.A2.combinew(i32 %flush, i32 %flush)
  %1 = bitcast i64 %0 to <2 x i32>
  br label %polly.loop_body

polly.loop_after:                                 ; preds = %polly.loop_body
  ret i32 0

polly.loop_body:                                  ; preds = %entry, %polly.loop_body
  %p_arrayidx35.phi = phi i16* [ %dstImg, %entry ], [ %p_arrayidx35.inc, %polly.loop_body ]
  %p_arrayidx.phi = phi i16* [ %src, %entry ], [ %p_arrayidx.inc, %polly.loop_body ]
  %polly.loopiv56 = phi i32 [ 0, %entry ], [ %polly.next_loopiv, %polly.loop_body ]
  %polly.next_loopiv = add nsw i32 %polly.loopiv56, 4
  %vector_ptr = bitcast i16* %p_arrayidx.phi to <4 x i16>*
  %_p_vec_full = load <4 x i16>, <4 x i16>* %vector_ptr, align 2
  %_high_half = shufflevector <4 x i16> %_p_vec_full, <4 x i16> undef, <2 x i32> <i32 2, i32 3>
  %_low_half = shufflevector <4 x i16> %_p_vec_full, <4 x i16> undef, <2 x i32> <i32 0, i32 1>
  %2 = zext <2 x i16> %_low_half to <2 x i32>
  %3 = zext <2 x i16> %_high_half to <2 x i32>
  %add33p_vec = add <2 x i32> %2, %1
  %add33p_vec48 = add <2 x i32> %3, %1
  %4 = trunc <2 x i32> %add33p_vec to <2 x i16>
  %5 = trunc <2 x i32> %add33p_vec48 to <2 x i16>
  %_combined_vec = shufflevector <2 x i16> %4, <2 x i16> %5, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %vector_ptr49 = bitcast i16* %p_arrayidx35.phi to <4 x i16>*
  store <4 x i16> %_combined_vec, <4 x i16>* %vector_ptr49, align 2
  %6 = icmp slt i32 %polly.next_loopiv, 1024
  %p_arrayidx35.inc = getelementptr i16, i16* %p_arrayidx35.phi, i32 4
  %p_arrayidx.inc = getelementptr i16, i16* %p_arrayidx.phi, i32 4
  br i1 %6, label %polly.loop_body, label %polly.loop_after
}

declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }


