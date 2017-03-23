; RUN: opt -mtriple=amdgcn-- -S -structurizecfg -si-annotate-control-flow %s | FileCheck -check-prefix=OPT %s
; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s


; OPT-LABEL: @annotate_unreachable_noloop(
; OPT-NOT: call i1 @llvm.amdgcn.loop

; GCN-LABEL: {{^}}annotate_unreachable_noloop:
; GCN: s_cbranch_vccnz
; GCN-NOT: s_endpgm
; GCN: .Lfunc_end0
define amdgpu_kernel void @annotate_unreachable_noloop(<4 x float> addrspace(1)* noalias nocapture readonly %arg) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %bb1

bb1:                                              ; preds = %bb
  %tmp2 = sext i32 %tmp to i64
  %tmp3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %arg, i64 %tmp2
  %tmp4 = load <4 x float>, <4 x float> addrspace(1)* %tmp3, align 16
  br i1 undef, label %bb5, label %bb3

bb3:                                              ; preds = %bb1
  %tmp6 = extractelement <4 x float> %tmp4, i32 2
  %tmp7 = fcmp olt float %tmp6, 0.000000e+00
  br i1 %tmp7, label %bb4, label %bb5 ; crash goes away if these are swapped

bb4:                                              ; preds = %bb3
  unreachable

bb5:                                              ; preds = %bb3, %bb1
  unreachable
}


; OPT-LABEL: @annotate_ret_noloop(
; OPT-NOT: call i1 @llvm.amdgcn.loop

; GCN-LABEL: {{^}}annotate_ret_noloop:
; GCN: s_cbranch_scc1
; GCN: s_endpgm
; GCN: .Lfunc_end1
define amdgpu_kernel void @annotate_ret_noloop(<4 x float> addrspace(1)* noalias nocapture readonly %arg) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %bb1

bb1:                                              ; preds = %bb
  %tmp2 = sext i32 %tmp to i64
  %tmp3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %arg, i64 %tmp2
  %tmp4 = load <4 x float>, <4 x float> addrspace(1)* %tmp3, align 16
  br i1 undef, label %bb5, label %bb3

bb3:                                              ; preds = %bb1
  %tmp6 = extractelement <4 x float> %tmp4, i32 2
  %tmp7 = fcmp olt float %tmp6, 0.000000e+00
  br i1 %tmp7, label %bb4, label %bb5 ; crash goes away if these are swapped

bb4:                                              ; preds = %bb3
  ret void

bb5:                                              ; preds = %bb3, %bb1
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
