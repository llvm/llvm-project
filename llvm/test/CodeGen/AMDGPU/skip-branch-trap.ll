; RUN: llc -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; FIXME: merge with trap.ll

; An s_cbranch_execnz is required to avoid trapping if all lanes are 0
; GCN-LABEL: {{^}}trap_divergent_branch:
; GCN: s_and_saveexec_b64
; GCN: s_cbranch_execnz [[TRAP:.LBB[0-9]+_[0-9]+]]
; GCN: ; %bb.{{[0-9]+}}:
; GCN-NEXT: s_endpgm
; GCN: [[TRAP]]:
; GCN: s_trap 2
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @trap_divergent_branch(i32 addrspace(1)* nocapture readonly %arg) {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %id
  %divergent.val = load i32, i32 addrspace(1)* %gep
  %cmp = icmp eq i32 %divergent.val, 0
  br i1 %cmp, label %bb, label %end

bb:
  call void @llvm.trap()
  br label %end

end:
  ret void
}

; GCN-LABEL: {{^}}debugtrap_divergent_branch:
; GCN: s_and_saveexec_b64
; GCN: s_cbranch_execz [[ENDPGM:.LBB[0-9]+_[0-9]+]]
; GCN: ; %bb.{{[0-9]+}}:
; GCN: s_trap 3
; GCN-NEXT: [[ENDPGM]]:
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @debugtrap_divergent_branch(i32 addrspace(1)* nocapture readonly %arg) {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %id
  %divergent.val = load i32, i32 addrspace(1)* %gep
  %cmp = icmp eq i32 %divergent.val, 0
  br i1 %cmp, label %bb, label %end

bb:
  call void @llvm.debugtrap()
  br label %end

end:
  ret void
}

declare void @llvm.trap() #0
declare void @llvm.debugtrap() #1
declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind noreturn }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone speculatable }
