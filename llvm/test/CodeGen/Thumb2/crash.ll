; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -verify-machineinstrs
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

; This function would crash LiveIntervalAnalysis by creating a chain of 4 INSERT_SUBREGs of the same register.
define arm_apcscc void @NEON_vst4q_u32(i32* nocapture %sp0, i32* nocapture %sp1, i32* nocapture %sp2, i32* nocapture %sp3, i32* %dp) nounwind {
entry:
  %0 = bitcast i32* %sp0 to <4 x i32>*            ; <<4 x i32>*> [#uses=1]
  %1 = load <4 x i32>* %0, align 16               ; <<4 x i32>> [#uses=1]
  %2 = bitcast i32* %sp1 to <4 x i32>*            ; <<4 x i32>*> [#uses=1]
  %3 = load <4 x i32>* %2, align 16               ; <<4 x i32>> [#uses=1]
  %4 = bitcast i32* %sp2 to <4 x i32>*            ; <<4 x i32>*> [#uses=1]
  %5 = load <4 x i32>* %4, align 16               ; <<4 x i32>> [#uses=1]
  %6 = bitcast i32* %sp3 to <4 x i32>*            ; <<4 x i32>*> [#uses=1]
  %7 = load <4 x i32>* %6, align 16               ; <<4 x i32>> [#uses=1]
  %8 = bitcast i32* %dp to i8*                    ; <i8*> [#uses=1]
  tail call void @llvm.arm.neon.vst4.v4i32(i8* %8, <4 x i32> %1, <4 x i32> %3, <4 x i32> %5, <4 x i32> %7, i32 1)
  ret void
}

declare void @llvm.arm.neon.vst4.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32) nounwind

@sbuf = common global [16 x i32] zeroinitializer, align 16 ; <[16 x i32]*> [#uses=5]
@dbuf = common global [16 x i32] zeroinitializer  ; <[16 x i32]*> [#uses=2]

; This function creates 4 chained INSERT_SUBREGS and then invokes the register scavenger.
; The first INSERT_SUBREG needs an <undef> use operand for that to work.
define arm_apcscc i32 @main() nounwind {
bb.nph:
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %0 = phi i32 [ 0, %bb.nph ], [ %1, %bb ]        ; <i32> [#uses=4]
  %scevgep = getelementptr [16 x i32]* @sbuf, i32 0, i32 %0 ; <i32*> [#uses=1]
  %scevgep5 = getelementptr [16 x i32]* @dbuf, i32 0, i32 %0 ; <i32*> [#uses=1]
  store i32 %0, i32* %scevgep, align 4
  store i32 -1, i32* %scevgep5, align 4
  %1 = add nsw i32 %0, 1                          ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %1, 16                  ; <i1> [#uses=1]
  br i1 %exitcond, label %bb2, label %bb

bb2:                                              ; preds = %bb
  %2 = load <4 x i32>* bitcast ([16 x i32]* @sbuf to <4 x i32>*), align 16 ; <<4 x i32>> [#uses=1]
  %3 = load <4 x i32>* bitcast (i32* getelementptr inbounds ([16 x i32]* @sbuf, i32 0, i32 4) to <4 x i32>*), align 16 ; <<4 x i32>> [#uses=1]
  %4 = load <4 x i32>* bitcast (i32* getelementptr inbounds ([16 x i32]* @sbuf, i32 0, i32 8) to <4 x i32>*), align 16 ; <<4 x i32>> [#uses=1]
  %5 = load <4 x i32>* bitcast (i32* getelementptr inbounds ([16 x i32]* @sbuf, i32 0, i32 12) to <4 x i32>*), align 16 ; <<4 x i32>> [#uses=1]
  tail call void @llvm.arm.neon.vst4.v4i32(i8* bitcast ([16 x i32]* @dbuf to i8*), <4 x i32> %2, <4 x i32> %3, <4 x i32> %4, <4 x i32> %5, i32 1) nounwind
  ret i32 0
}

; PR12389
; Make sure the DPair register class can spill.
define void @pr12389(i8* %p) nounwind ssp {
entry:
  %vld1 = tail call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* %p, i32 1)
  tail call void asm sideeffect "", "~{q0},~{q1},~{q2},~{q3},~{q4},~{q5},~{q6},~{q7},~{q8},~{q9},~{q10},~{q11},~{q12},~{q13},~{q14},~{q15}"() nounwind
  tail call void @llvm.arm.neon.vst1.v4f32(i8* %p, <4 x float> %vld1, i32 1)
  ret void
}

declare <4 x float> @llvm.arm.neon.vld1.v4f32(i8*, i32) nounwind readonly

declare void @llvm.arm.neon.vst1.v4f32(i8*, <4 x float>, i32) nounwind

; <rdar://problem/11101911>
; When an strd is expanded into two str instructions, make sure the first str
; doesn't kill the base register. This can happen if the base register is the
; same as the data register.
%class = type { i8*, %class*, i32 }
define void @f11101911(%class* %this, i32 %num) ssp align 2 {
entry:
  %p1 = getelementptr inbounds %class* %this, i32 0, i32 1
  %p2 = getelementptr inbounds %class* %this, i32 0, i32 2
  tail call void asm sideeffect "", "~{r1},~{r3},~{r5},~{r11},~{r13}"() nounwind
  store %class* %this, %class** %p1, align 4
  store i32 %num, i32* %p2, align 4
  ret void
}
