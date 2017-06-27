; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}alignbit_shr_pat:
; GCN-DAG: s_load_dword s[[SHR:[0-9]+]]
; GCN-DAG: load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; GCN: v_alignbit_b32 v{{[0-9]+}}, v[[HI]], v[[LO]], s[[SHR]]

define amdgpu_kernel void @alignbit_shr_pat(i64 addrspace(1)* nocapture readonly %arg, i32 addrspace(1)* nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, i64 addrspace(1)* %arg, align 8
  %tmp3 = and i32 %arg2, 31
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = lshr i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, i32 addrspace(1)* %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shl_pat:
; GCN-DAG: s_load_dword s[[SHL:[0-9]+]]
; GCN-DAG: load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; GCN-DAG: s_sub_i32 s[[SHR:[0-9]+]], 32, s[[SHL]]
; GCN:     v_alignbit_b32 v{{[0-9]+}}, v[[HI]], v[[LO]], s[[SHR]]

define amdgpu_kernel void @alignbit_shl_pat(i64 addrspace(1)* nocapture readonly %arg, i32 addrspace(1)* nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, i64 addrspace(1)* %arg, align 8
  %tmp3 = and i32 %arg2, 31
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = shl i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, i32 addrspace(1)* %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shr_pat_v:
; GCN-DAG: load_dword v[[SHR:[0-9]+]],
; GCN-DAG: load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; GCN: v_alignbit_b32 v{{[0-9]+}}, v[[HI]], v[[LO]], v[[SHR]]

define amdgpu_kernel void @alignbit_shr_pat_v(i64 addrspace(1)* nocapture readonly %arg, i32 addrspace(1)* nocapture %arg1) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep1 = getelementptr inbounds i64, i64 addrspace(1)* %arg, i32 %tid
  %tmp = load i64, i64 addrspace(1)* %gep1, align 8
  %gep2 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i32 %tid
  %amt = load i32, i32 addrspace(1)* %gep2, align 4
  %tmp3 = and i32 %amt, 31
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = lshr i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, i32 addrspace(1)* %gep2, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shl_pat_v:
; GCN-DAG: load_dword v[[SHL:[0-9]+]],
; GCN-DAG: load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; GCN-DAG: v_sub_i32_e32 v[[SHR:[0-9]+]], {{[^,]+}}, 32, v[[SHL]]
; GCN: v_alignbit_b32 v{{[0-9]+}}, v[[HI]], v[[LO]], v[[SHR]]

define amdgpu_kernel void @alignbit_shl_pat_v(i64 addrspace(1)* nocapture readonly %arg, i32 addrspace(1)* nocapture %arg1) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep1 = getelementptr inbounds i64, i64 addrspace(1)* %arg, i32 %tid
  %tmp = load i64, i64 addrspace(1)* %gep1, align 8
  %gep2 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i32 %tid
  %amt = load i32, i32 addrspace(1)* %gep2, align 4
  %tmp3 = and i32 %amt, 31
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = shl i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, i32 addrspace(1)* %gep2, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shr_pat_wrong_and30:
; Negative test, wrong constant
; GCN: v_lshr_b64
; GCN-NOT: v_alignbit_b32

define amdgpu_kernel void @alignbit_shr_pat_wrong_and30(i64 addrspace(1)* nocapture readonly %arg, i32 addrspace(1)* nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, i64 addrspace(1)* %arg, align 8
  %tmp3 = and i32 %arg2, 30
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = lshr i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, i32 addrspace(1)* %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shl_pat_wrong_and30:
; Negative test, wrong constant
; GCN: v_lshl_b64
; GCN-NOT: v_alignbit_b32

define amdgpu_kernel void @alignbit_shl_pat_wrong_and30(i64 addrspace(1)* nocapture readonly %arg, i32 addrspace(1)* nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, i64 addrspace(1)* %arg, align 8
  %tmp3 = and i32 %arg2, 30
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = shl i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, i32 addrspace(1)* %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shr_pat_wrong_and63:
; Negative test, wrong constant
; GCN: v_lshr_b64
; GCN-NOT: v_alignbit_b32

define amdgpu_kernel void @alignbit_shr_pat_wrong_and63(i64 addrspace(1)* nocapture readonly %arg, i32 addrspace(1)* nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, i64 addrspace(1)* %arg, align 8
  %tmp3 = and i32 %arg2, 63
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = lshr i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, i32 addrspace(1)* %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}alignbit_shl_pat_wrong_and63:
; Negative test, wrong constant
; GCN: v_lshl_b64
; GCN-NOT: v_alignbit_b32

define amdgpu_kernel void @alignbit_shl_pat_wrong_and63(i64 addrspace(1)* nocapture readonly %arg, i32 addrspace(1)* nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, i64 addrspace(1)* %arg, align 8
  %tmp3 = and i32 %arg2, 63
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = shl i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, i32 addrspace(1)* %arg1, align 4
  ret void
}
declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone speculatable }
