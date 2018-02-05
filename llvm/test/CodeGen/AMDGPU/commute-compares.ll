; RUN: llc -march=amdgcn -amdgpu-sdwa-peephole=0 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s

declare i32 @llvm.amdgcn.workitem.id.x() #0

; --------------------------------------------------------------------------------
; i32 compares
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}commute_eq_64_i32:
; GCN: v_cmp_eq_u32_e32 vcc, 64, v{{[0-9]+}}
define amdgpu_kernel void @commute_eq_64_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp eq i32 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ne_64_i32:
; GCN: v_cmp_ne_u32_e32 vcc, 64, v{{[0-9]+}}
define amdgpu_kernel void @commute_ne_64_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp ne i32 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; FIXME: Why isn't this being folded as a constant?
; GCN-LABEL: {{^}}commute_ne_litk_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0x3039
; GCN: v_cmp_ne_u32_e32 vcc, v{{[0-9]+}}, [[K]]
define amdgpu_kernel void @commute_ne_litk_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp ne i32 %val, 12345
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ugt_64_i32:
; GCN: v_cmp_lt_u32_e32 vcc, 64, v{{[0-9]+}}
define amdgpu_kernel void @commute_ugt_64_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp ugt i32 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_uge_64_i32:
; GCN: v_cmp_lt_u32_e32 vcc, 63, v{{[0-9]+}}
define amdgpu_kernel void @commute_uge_64_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp uge i32 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ult_64_i32:
; GCN: v_cmp_gt_u32_e32 vcc, 64, v{{[0-9]+}}
define amdgpu_kernel void @commute_ult_64_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp ult i32 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ule_63_i32:
; GCN: v_cmp_gt_u32_e32 vcc, 64, v{{[0-9]+}}
define amdgpu_kernel void @commute_ule_63_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp ule i32 %val, 63
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ule_64_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0x41{{$}}
; GCN: v_cmp_lt_u32_e32 vcc, v{{[0-9]+}}, [[K]]
define amdgpu_kernel void @commute_ule_64_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp ule i32 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_sgt_neg1_i32:
; GCN: v_cmp_lt_i32_e32 vcc, -1, v{{[0-9]+}}
define amdgpu_kernel void @commute_sgt_neg1_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp sgt i32 %val, -1
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_sge_neg2_i32:
; GCN: v_cmp_lt_i32_e32 vcc, -3, v{{[0-9]+}}
define amdgpu_kernel void @commute_sge_neg2_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp sge i32 %val, -2
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_slt_neg16_i32:
; GCN: v_cmp_gt_i32_e32 vcc, -16, v{{[0-9]+}}
define amdgpu_kernel void @commute_slt_neg16_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp slt i32 %val, -16
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_sle_5_i32:
; GCN: v_cmp_gt_i32_e32 vcc, 6, v{{[0-9]+}}
define amdgpu_kernel void @commute_sle_5_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep.in
  %cmp = icmp sle i32 %val, 5
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; --------------------------------------------------------------------------------
; i64 compares
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}commute_eq_64_i64:
; GCN: v_cmp_eq_u64_e32 vcc, 64, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_eq_64_i64(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep.in
  %cmp = icmp eq i64 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ne_64_i64:
; GCN: v_cmp_ne_u64_e32 vcc, 64, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_ne_64_i64(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep.in
  %cmp = icmp ne i64 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ugt_64_i64:
; GCN: v_cmp_lt_u64_e32 vcc, 64, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_ugt_64_i64(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep.in
  %cmp = icmp ugt i64 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_uge_64_i64:
; GCN: v_cmp_lt_u64_e32 vcc, 63, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_uge_64_i64(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep.in
  %cmp = icmp uge i64 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ult_64_i64:
; GCN: v_cmp_gt_u64_e32 vcc, 64, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_ult_64_i64(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep.in
  %cmp = icmp ult i64 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ule_63_i64:
; GCN: v_cmp_gt_u64_e32 vcc, 64, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_ule_63_i64(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep.in
  %cmp = icmp ule i64 %val, 63
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; FIXME: Undo canonicalization to gt (x + 1) since it doesn't use the inline imm

; GCN-LABEL: {{^}}commute_ule_64_i64:
; GCN-DAG: s_movk_i32 s[[KLO:[0-9]+]], 0x41{{$}}
; GCN: v_cmp_gt_u64_e32 vcc, s{{\[}}[[KLO]]:{{[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_ule_64_i64(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep.in
  %cmp = icmp ule i64 %val, 64
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_sgt_neg1_i64:
; GCN: v_cmp_lt_i64_e32 vcc, -1, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_sgt_neg1_i64(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep.in
  %cmp = icmp sgt i64 %val, -1
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_sge_neg2_i64:
; GCN: v_cmp_lt_i64_e32 vcc, -3, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_sge_neg2_i64(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep.in
  %cmp = icmp sge i64 %val, -2
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_slt_neg16_i64:
; GCN: v_cmp_gt_i64_e32 vcc, -16, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_slt_neg16_i64(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep.in
  %cmp = icmp slt i64 %val, -16
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_sle_5_i64:
; GCN: v_cmp_gt_i64_e32 vcc, 6, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_sle_5_i64(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep.in
  %cmp = icmp sle i64 %val, 5
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; --------------------------------------------------------------------------------
; f32 compares
; --------------------------------------------------------------------------------


; GCN-LABEL: {{^}}commute_oeq_2.0_f32:
; GCN: v_cmp_eq_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_oeq_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp oeq float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}


; GCN-LABEL: {{^}}commute_ogt_2.0_f32:
; GCN: v_cmp_lt_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_ogt_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp ogt float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_oge_2.0_f32:
; GCN: v_cmp_le_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_oge_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp oge float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_olt_2.0_f32:
; GCN: v_cmp_gt_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_olt_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp olt float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ole_2.0_f32:
; GCN: v_cmp_ge_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_ole_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp ole float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_one_2.0_f32:
; GCN: v_cmp_lg_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_one_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp one float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ord_2.0_f32:
; GCN: v_cmp_o_f32_e32 vcc, [[REG:v[0-9]+]], [[REG]]
define amdgpu_kernel void @commute_ord_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp ord float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ueq_2.0_f32:
; GCN: v_cmp_nlg_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_ueq_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp ueq float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ugt_2.0_f32:
; GCN: v_cmp_nge_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_ugt_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp ugt float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_uge_2.0_f32:
; GCN: v_cmp_ngt_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_uge_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp uge float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ult_2.0_f32:
; GCN: v_cmp_nle_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_ult_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp ult float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ule_2.0_f32:
; GCN: v_cmp_nlt_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_ule_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp ule float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_une_2.0_f32:
; GCN: v_cmp_neq_f32_e32 vcc, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @commute_une_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp une float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_uno_2.0_f32:
; GCN: v_cmp_u_f32_e32 vcc, [[REG:v[0-9]+]], [[REG]]
define amdgpu_kernel void @commute_uno_2.0_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load float, float addrspace(1)* %gep.in
  %cmp = fcmp uno float %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; --------------------------------------------------------------------------------
; f64 compares
; --------------------------------------------------------------------------------


; GCN-LABEL: {{^}}commute_oeq_2.0_f64:
; GCN: v_cmp_eq_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_oeq_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp oeq double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}


; GCN-LABEL: {{^}}commute_ogt_2.0_f64:
; GCN: v_cmp_lt_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_ogt_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp ogt double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_oge_2.0_f64:
; GCN: v_cmp_le_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_oge_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp oge double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_olt_2.0_f64:
; GCN: v_cmp_gt_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_olt_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp olt double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ole_2.0_f64:
; GCN: v_cmp_ge_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_ole_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp ole double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_one_2.0_f64:
; GCN: v_cmp_lg_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_one_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp one double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ord_2.0_f64:
; GCN: v_cmp_o_f64_e32 vcc, [[REG:v\[[0-9]+:[0-9]+\]]], [[REG]]
define amdgpu_kernel void @commute_ord_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp ord double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ueq_2.0_f64:
; GCN: v_cmp_nlg_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_ueq_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp ueq double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ugt_2.0_f64:
; GCN: v_cmp_nge_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_ugt_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp ugt double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_uge_2.0_f64:
; GCN: v_cmp_ngt_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_uge_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp uge double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ult_2.0_f64:
; GCN: v_cmp_nle_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_ult_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp ult double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_ule_2.0_f64:
; GCN: v_cmp_nlt_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_ule_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp ule double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_une_2.0_f64:
; GCN: v_cmp_neq_f64_e32 vcc, 2.0, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @commute_une_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp une double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}commute_uno_2.0_f64:
; GCN: v_cmp_u_f64_e32 vcc, [[REG:v\[[0-9]+:[0-9]+\]]], [[REG]]
define amdgpu_kernel void @commute_uno_2.0_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load double, double addrspace(1)* %gep.in
  %cmp = fcmp uno double %val, 2.0
  %ext = sext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %gep.out
  ret void
}


; FIXME: Should be able to fold this frameindex
; Without commuting the frame index in the pre-regalloc run of
; SIShrinkInstructions, this was using the VOP3 compare.

; GCN-LABEL: {{^}}commute_frameindex:
; XGCN: v_cmp_eq_u32_e32 vcc, 0, v{{[0-9]+}}

; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4{{$}}
; GCN: v_cmp_eq_u32_e32 vcc, v{{[0-9]+}}, [[FI]]
define amdgpu_kernel void @commute_frameindex(i32 addrspace(1)* nocapture %out) #0 {
entry:
  %stack0 = alloca i32, addrspace(5)
  %ptr0 = load volatile i32 addrspace(5)*, i32 addrspace(5)* addrspace(1)* undef
  %eq = icmp eq i32 addrspace(5)* %ptr0, %stack0
  %ext = zext i1 %eq to i32
  store volatile i32 %ext, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
