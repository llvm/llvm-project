; RUN: opt -mtriple amdgcn-- -passes='print<divergence>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT:  %tmp5 = getelementptr inbounds float, ptr addrspace(1) %arg, i64 %tmp2
; CHECK: DIVERGENT:  %tmp10 = load volatile float, ptr addrspace(1) %tmp5, align 4
; CHECK: DIVERGENT:  %tmp11 = load volatile float, ptr addrspace(1) %tmp5, align 4

; The post dominator tree does not have a root node in this case
define amdgpu_kernel void @no_return_blocks(ptr addrspace(1) noalias nocapture readonly %arg, ptr addrspace(1) noalias nocapture readonly %arg1) #0 {
bb0:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tmp2 = sext i32 %tmp to i64
  %tmp5 = getelementptr inbounds float, ptr addrspace(1) %arg, i64 %tmp2
  %tmp6 = load volatile float, ptr addrspace(1) %tmp5, align 4
  %tmp8 = fcmp olt float %tmp6, 0.000000e+00
  br i1 %tmp8, label %bb1, label %bb2

bb1:
  %tmp10 = load volatile float, ptr addrspace(1) %tmp5, align 4
  br label %bb2

bb2:
  %tmp11 = load volatile float, ptr addrspace(1) %tmp5, align 4
  br label %bb1
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
