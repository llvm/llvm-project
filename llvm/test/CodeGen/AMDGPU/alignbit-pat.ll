; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}alignbit_shr_pat:
; GCN-DAG: s_load_dword s[[SHR:[0-9]+]]
; GCN-DAG: load_dwordx2 v[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; GCN: v_alignbit_b32 v{{[0-9]+}}, v[[HI]], v[[LO]], s[[SHR]]

define amdgpu_kernel void @alignbit_shr_pat(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, ptr addrspace(1) %arg, align 8
  %tmp3 = and i32 %arg2, 31
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = lshr i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, ptr addrspace(1) %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shr_pat_v:
; GCN-DAG: load_dword v[[SHR:[0-9]+]],
; GCN-DAG: load_dwordx2 v[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; GCN: v_alignbit_b32 v{{[0-9]+}}, v[[HI]], v[[LO]], v[[SHR]]

define amdgpu_kernel void @alignbit_shr_pat_v(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep1 = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %tid
  %tmp = load i64, ptr addrspace(1) %gep1, align 8
  %gep2 = getelementptr inbounds i32, ptr addrspace(1) %arg1, i32 %tid
  %amt = load i32, ptr addrspace(1) %gep2, align 4
  %tmp3 = and i32 %amt, 31
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = lshr i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, ptr addrspace(1) %gep2, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shr_pat_wrong_and30:
; Negative test, wrong constant
; GCN: v_lshr_b64
; GCN-NOT: v_alignbit_b32

define amdgpu_kernel void @alignbit_shr_pat_wrong_and30(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, ptr addrspace(1) %arg, align 8
  %tmp3 = and i32 %arg2, 30
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = lshr i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, ptr addrspace(1) %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shr_pat_wrong_and63:
; Negative test, wrong constant
; GCN: v_lshr_b64
; GCN-NOT: v_alignbit_b32

define amdgpu_kernel void @alignbit_shr_pat_wrong_and63(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, ptr addrspace(1) %arg, align 8
  %tmp3 = and i32 %arg2, 63
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = lshr i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, ptr addrspace(1) %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shr_pat_const30:
; GCN: load_dwordx2 v[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; GCN: v_alignbit_b32 v{{[0-9]+}}, v[[HI]], v[[LO]], 30

define amdgpu_kernel void @alignbit_shr_pat_const30(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1) {
bb:
  %tmp = load i64, ptr addrspace(1) %arg, align 8
  %tmp5 = lshr i64 %tmp, 30
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, ptr addrspace(1) %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shr_pat_wrong_const33:
; Negative test, shift amount more than 31
; GCN: v_lshrrev_b32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}
; GCN-NOT: v_alignbit_b32

define amdgpu_kernel void @alignbit_shr_pat_wrong_const33(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1) {
bb:
  %tmp = load i64, ptr addrspace(1) %arg, align 8
  %tmp5 = lshr i64 %tmp, 33
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, ptr addrspace(1) %arg1, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone speculatable }
