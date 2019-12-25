; RUN: opt < %s -slsr -S | FileCheck %s

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--"

%struct.Matrix4x4 = type { [4 x [4 x float]] }

; Function Attrs: nounwind
define fastcc void @Accelerator_Intersect(%struct.Matrix4x4 addrspace(1)* nocapture readonly %leafTransformations) #0 {
; CHECK-LABEL:  @Accelerator_Intersect(
entry:
  %tmp = sext i32 undef to i64
  %arrayidx114 = getelementptr inbounds %struct.Matrix4x4, %struct.Matrix4x4 addrspace(1)* %leafTransformations, i64 %tmp
  %tmp1 = getelementptr %struct.Matrix4x4, %struct.Matrix4x4 addrspace(1)* %leafTransformations, i64 %tmp, i32 0, i64 0, i64 0
; CHECK: %tmp1 = getelementptr %struct.Matrix4x4, %struct.Matrix4x4 addrspace(1)* %leafTransformations, i64 %tmp, i32 0, i64 0, i64 0
  %tmp2 = load <4 x float>, <4 x float> addrspace(1)* undef, align 4
  ret void
}

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "target-cpu"="tahiti" "unsafe-fp-math"="false" "use-soft-float"="false" }
