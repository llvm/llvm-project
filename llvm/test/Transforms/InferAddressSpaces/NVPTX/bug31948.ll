; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -infer-address-spaces %s | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

%struct.bar = type { float, float* }

@var1 = local_unnamed_addr addrspace(3) externally_initialized global %struct.bar undef, align 8

; CHECK-LABEL: @bug31948(
; CHECK: %tmp = load float*, float* addrspace(3)* getelementptr inbounds (%struct.bar, %struct.bar addrspace(3)* @var1, i64 0, i32 1), align 8
; CHECK: %tmp1 = load float, float* %tmp, align 4
; CHECK: store float %conv1, float* %tmp, align 4
; CHECK: store i32 32, i32 addrspace(3)* addrspacecast (i32* bitcast (float** getelementptr (%struct.bar, %struct.bar* addrspacecast (%struct.bar addrspace(3)* @var1 to %struct.bar*), i64 0, i32 1) to i32*) to i32 addrspace(3)*), align 4
define void @bug31948(float %a, float* nocapture readnone %x, float* nocapture readnone %y) local_unnamed_addr #0 {
entry:
  %tmp = load float*, float** getelementptr (%struct.bar, %struct.bar* addrspacecast (%struct.bar addrspace(3)* @var1 to %struct.bar*), i64 0, i32 1), align 8
  %tmp1 = load float, float* %tmp, align 4
  %conv1 = fadd float %tmp1, 1.000000e+00
  store float %conv1, float* %tmp, align 4
  store i32 32, i32* bitcast (float** getelementptr (%struct.bar, %struct.bar* addrspacecast (%struct.bar addrspace(3)* @var1 to %struct.bar*), i64 0, i32 1) to i32*), align 4
  ret void
}

attributes #0 = { norecurse nounwind }
