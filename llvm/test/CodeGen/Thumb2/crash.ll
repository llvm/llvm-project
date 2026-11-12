; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -verify-machineinstrs
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -verify-machineinstrs -O0
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

; This function would crash LiveIntervalAnalysis by creating a chain of 4 INSERT_SUBREGs of the same register.
define arm_apcscc void @NEON_vst4q_u32(ptr nocapture %sp0, ptr nocapture %sp1, ptr nocapture %sp2, ptr nocapture %sp3, ptr %dp) nounwind {
entry:
  %0 = load <4 x i32>, ptr %sp0, align 16               ; <<4 x i32>> [#uses=1]
  %1 = load <4 x i32>, ptr %sp1, align 16               ; <<4 x i32>> [#uses=1]
  %2 = load <4 x i32>, ptr %sp2, align 16               ; <<4 x i32>> [#uses=1]
  %3 = load <4 x i32>, ptr %sp3, align 16               ; <<4 x i32>> [#uses=1]
  tail call void @llvm.arm.neon.vst4.p0.v4i32(ptr %dp, <4 x i32> %0, <4 x i32> %1, <4 x i32> %2, <4 x i32> %3, i32 1)
  ret void
}

declare void @llvm.arm.neon.vst4.p0.v4i32(ptr, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32) nounwind

@sbuf = common global [16 x i32] zeroinitializer, align 16 ; <ptr> [#uses=5]
@dbuf = common global [16 x i32] zeroinitializer  ; <ptr> [#uses=2]

; This function creates 4 chained INSERT_SUBREGS and then invokes the register scavenger.
; The first INSERT_SUBREG needs an <undef> use operand for that to work.
define arm_apcscc i32 @main() nounwind {
bb.nph:
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %0 = phi i32 [ 0, %bb.nph ], [ %1, %bb ]        ; <i32> [#uses=4]
  %scevgep = getelementptr [16 x i32], ptr @sbuf, i32 0, i32 %0 ; <ptr> [#uses=1]
  %scevgep5 = getelementptr [16 x i32], ptr @dbuf, i32 0, i32 %0 ; <ptr> [#uses=1]
  store i32 %0, ptr %scevgep, align 4
  store i32 -1, ptr %scevgep5, align 4
  %1 = add nsw i32 %0, 1                          ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %1, 16                  ; <i1> [#uses=1]
  br i1 %exitcond, label %bb2, label %bb

bb2:                                              ; preds = %bb
  %2 = load <4 x i32>, ptr @sbuf, align 16 ; <<4 x i32>> [#uses=1]
  %3 = load <4 x i32>, ptr getelementptr inbounds ([16 x i32], ptr @sbuf, i32 0, i32 4), align 16 ; <<4 x i32>> [#uses=1]
  %4 = load <4 x i32>, ptr getelementptr inbounds ([16 x i32], ptr @sbuf, i32 0, i32 8), align 16 ; <<4 x i32>> [#uses=1]
  %5 = load <4 x i32>, ptr getelementptr inbounds ([16 x i32], ptr @sbuf, i32 0, i32 12), align 16 ; <<4 x i32>> [#uses=1]
  tail call void @llvm.arm.neon.vst4.p0.v4i32(ptr @dbuf, <4 x i32> %2, <4 x i32> %3, <4 x i32> %4, <4 x i32> %5, i32 1) nounwind
  ret i32 0
}

; PR12389
; Make sure the DPair register class can spill.
define void @pr12389(ptr %p) nounwind ssp {
entry:
  %vld1 = tail call <4 x float> @llvm.arm.neon.vld1.v4f32.p0(ptr %p, i32 1)
  tail call void asm sideeffect "", "~{q0},~{q1},~{q2},~{q3},~{q4},~{q5},~{q6},~{q7},~{q8},~{q9},~{q10},~{q11},~{q12},~{q13},~{q14},~{q15}"() nounwind
  tail call void @llvm.arm.neon.vst1.p0.v4f32(ptr %p, <4 x float> %vld1, i32 1)
  ret void
}

declare <4 x float> @llvm.arm.neon.vld1.v4f32.p0(ptr, i32) nounwind readonly

declare void @llvm.arm.neon.vst1.p0.v4f32(ptr, <4 x float>, i32) nounwind

; <rdar://problem/11101911>
; When an strd is expanded into two str instructions, make sure the first str
; doesn't kill the base register. This can happen if the base register is the
; same as the data register.
%class = type { ptr, ptr, i32 }
define void @f11101911(ptr %this, i32 %num) ssp align 2 {
entry:
  %p1 = getelementptr inbounds %class, ptr %this, i32 0, i32 1
  %p2 = getelementptr inbounds %class, ptr %this, i32 0, i32 2
  tail call void asm sideeffect "", "~{r1},~{r3},~{r5},~{r11},~{r13}"() nounwind
  store ptr %this, ptr %p1, align 4
  store i32 %num, ptr %p2, align 4
  ret void
}

; Check RAFast handling of inline assembly with many dense clobbers.
; The large tuple aliases of the vector registers can cause problems.
define void @rdar13249625(ptr nocapture %p) nounwind {
  %1 = tail call double asm sideeffect "@ $0", "=w,~{d0},~{q1},~{q2},~{q3},~{q4},~{q5},~{q6},~{q7},~{q8},~{q9},~{q10},~{q11},~{q12},~{q13},~{q14},~{q15}"() nounwind
  store double %1, ptr %p, align 4
  ret void
}
