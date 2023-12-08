; RUN: llc < %s -mcpu=cortex-a53 -enable-post-misched=false -enable-aa-sched-mi | FileCheck %s

; Check that the vector store intrinsic does not prevent fmla instructions from
; being scheduled together.  Since the vector loads and stores generated from
; the intrinsics do not alias each other, the store can be pushed past the load.
; This allows fmla instructions to be scheduled together.


; CHECK: fmla
; CHECK-NEXT: mov
; CHECK-NEXT: mov
; CHECK-NEXT: fmla
; CHECK-NEXT: fmla
; CHECK-NEXT: fmla
target datalayout = "e-m:e-i64:64-i128:128-n8:16:32:64-S128"
target triple = "aarch64--linux-gnu"

%Struct = type { ptr, [9 x double], [16 x {float, float}], [16 x {float, float}], i32, i32 }

; Function Attrs: nounwind
define linkonce_odr void @func(ptr nocapture %this, <4 x float> %f) unnamed_addr #0 align 2 {
entry:
  %scevgep = getelementptr %Struct, ptr %this, i64 0, i32 2, i64 8, i32 0
  %vec1 = tail call { <4 x float>, <4 x float> } @llvm.aarch64.neon.ld2.v4f32.p0(ptr %scevgep)
  %ev1 = extractvalue { <4 x float>, <4 x float> } %vec1, 1
  %fm1 = fmul <4 x float> %f, %ev1
  %av1 = fadd <4 x float> %f, %fm1
  %ev2 = extractvalue { <4 x float>, <4 x float> } %vec1, 0
  %fm2 = fmul <4 x float> %f, %ev2
  %av2 = fadd <4 x float> %f, %fm2
  %scevgep2 = getelementptr %Struct, ptr %this, i64 0, i32 3, i64 8, i32 0
  tail call void @llvm.aarch64.neon.st2.v4f32.p0(<4 x float> %av2, <4 x float> %av1, ptr %scevgep2)
  %scevgep3 = getelementptr %Struct, ptr %this, i64 0, i32 2, i64 12, i32 0
  %vec2 = tail call { <4 x float>, <4 x float> } @llvm.aarch64.neon.ld2.v4f32.p0(ptr %scevgep3)
  %ev3 = extractvalue { <4 x float>, <4 x float> } %vec2, 1
  %fm3 = fmul <4 x float> %f, %ev3
  %av3 = fadd <4 x float> %f, %fm3
  %ev4 = extractvalue { <4 x float>, <4 x float> } %vec2, 0
  %fm4 = fmul <4 x float> %f, %ev4
  %av4 = fadd <4 x float> %f, %fm4
  %scevgep4 = getelementptr %Struct, ptr %this, i64 0, i32 3, i64 12, i32 0
  tail call void @llvm.aarch64.neon.st2.v4f32.p0(<4 x float> %av4, <4 x float> %av3, ptr %scevgep4)
  ret void
}

; Function Attrs: nounwind readonly
declare { <4 x float>, <4 x float> } @llvm.aarch64.neon.ld2.v4f32.p0(ptr) #2

; Function Attrs: nounwind
declare void @llvm.aarch64.neon.st2.v4f32.p0(<4 x float>, <4 x float>, ptr nocapture) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { nounwind readonly }
