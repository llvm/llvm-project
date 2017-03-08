; RUN: opt -mtriple=amdgcn-- -S -structurizecfg -si-annotate-control-flow %s | FileCheck -check-prefix=OPT %s
; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s


; OPT-LABEL: @annotate_unreachable(
; OPT: call { i1, i64 } @llvm.amdgcn.if(
; OPT-NOT: call void @llvm.amdgcn.end.cf(


; GCN-LABEL: {{^}}annotate_unreachable:
; GCN: s_and_saveexec_b64
; GCN-NOT: s_endpgm
; GCN: .Lfunc_end0
define void @annotate_unreachable(<4 x float> addrspace(1)* noalias nocapture readonly %arg) #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %bb1

bb1:                                              ; preds = %bb
  %tmp2 = sext i32 %tmp to i64
  %tmp3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %arg, i64 %tmp2
  %tmp4 = load <4 x float>, <4 x float> addrspace(1)* %tmp3, align 16
  br i1 undef, label %bb3, label %bb5  ; label order reversed

bb3:                                              ; preds = %bb1
  %tmp6 = extractelement <4 x float> %tmp4, i32 2
  %tmp7 = fcmp olt float %tmp6, 0.000000e+00
  br i1 %tmp7, label %bb4, label %bb5

bb4:                                              ; preds = %bb3
  unreachable

bb5:                                              ; preds = %bb3, %bb1
  unreachable
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
