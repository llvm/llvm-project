; RUN: opt < %s -passes=slsr -S | FileCheck %s
; RUN: opt < %s -passes='slsr' -S | FileCheck %s

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--"

%struct.Matrix4x4 = type { [4 x [4 x float]] }

; Function Attrs: nounwind
define fastcc void @Accelerator_Intersect(ptr addrspace(1) nocapture readonly %leafTransformations) {
; CHECK-LABEL:  @Accelerator_Intersect(
entry:
  %tmp = sext i32 undef to i64
  %arrayidx114 = getelementptr inbounds %struct.Matrix4x4, ptr addrspace(1) %leafTransformations, i64 %tmp
  %tmp1 = getelementptr %struct.Matrix4x4, ptr addrspace(1) %leafTransformations, i64 %tmp, i32 0, i64 0, i64 1
; CHECK: %tmp1 =  getelementptr i8, ptr addrspace(1) %arrayidx114, i64 4
  %tmp2 = load <4 x float>, ptr addrspace(1) undef, align 4
  ret void
}
