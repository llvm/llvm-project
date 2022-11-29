; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes=infer-address-spaces %s | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

%struct.bar = type { float, ptr }

@var1 = local_unnamed_addr addrspace(3) externally_initialized global %struct.bar undef, align 8

; CHECK-LABEL: @bug31948(
; CHECK: %tmp = load ptr, ptr addrspace(3) getelementptr inbounds (%struct.bar, ptr addrspace(3) @var1, i64 0, i32 1), align 8
; CHECK: %tmp1 = load float, ptr %tmp, align 4
; CHECK: store float %conv1, ptr %tmp, align 4
; CHECK: store i32 32, ptr addrspace(3) getelementptr inbounds (%struct.bar, ptr addrspace(3) @var1, i64 0, i32 1), align 4
define void @bug31948(float %a, ptr nocapture readnone %x, ptr nocapture readnone %y) local_unnamed_addr #0 {
entry:
  %tmp = load ptr, ptr getelementptr (%struct.bar, ptr addrspacecast (ptr addrspace(3) @var1 to ptr), i64 0, i32 1), align 8
  %tmp1 = load float, ptr %tmp, align 4
  %conv1 = fadd float %tmp1, 1.000000e+00
  store float %conv1, ptr %tmp, align 4
  store i32 32, ptr bitcast (ptr getelementptr (%struct.bar, ptr addrspacecast (ptr addrspace(3) @var1 to ptr), i64 0, i32 1) to ptr), align 4
  ret void
}

attributes #0 = { norecurse nounwind }
