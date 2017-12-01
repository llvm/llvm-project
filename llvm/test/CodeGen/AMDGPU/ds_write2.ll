; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -mattr=+load-store-opt < %s | FileCheck -enable-var-scope -strict-whitespace -check-prefixes=GCN,CI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -mattr=+load-store-opt,+flat-for-global < %s | FileCheck -enable-var-scope -strict-whitespace -check-prefixes=GCN,GFX9 %s

@lds = addrspace(3) global [512 x float] undef, align 4
@lds.f64 = addrspace(3) global [512 x double] undef, align 8

; GCN-LABEL: {{^}}simple_write2_one_val_f32:
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: {{buffer|flat|global}}_load_dword [[VAL:v[0-9]+]]
; GCN-DAG: v_lshlrev_b32_e32 [[VPTR:v[0-9]+]], 2, v{{[0-9]+}}
; GCN: ds_write2_b32 [[VPTR]], [[VAL]], [[VAL]] offset1:8
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_one_val_f32(float addrspace(1)* %C, float addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in.gep = getelementptr float, float addrspace(1)* %in, i32 %x.i
  %val = load float, float addrspace(1)* %in.gep, align 4
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  store float %val, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  store float %val, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; GCN-LABEL: {{^}}simple_write2_two_val_f32:
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; CI-DAG: buffer_load_dword [[VAL0:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CI-DAG: buffer_load_dword [[VAL1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4

; GFX9-DAG: global_load_dword [[VAL0:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, off{{$}}
; GFX9-DAG: global_load_dword [[VAL1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, off offset:4

; GCN-DAG: v_lshlrev_b32_e32 [[VPTR:v[0-9]+]], 2, v{{[0-9]+}}
; GCN: ds_write2_b32 [[VPTR]], [[VAL0]], [[VAL1]] offset1:8
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_two_val_f32(float addrspace(1)* %C, float addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in.gep.0 = getelementptr float, float addrspace(1)* %in, i32 %x.i
  %in.gep.1 = getelementptr float, float addrspace(1)* %in.gep.0, i32 1
  %val0 = load volatile float, float addrspace(1)* %in.gep.0, align 4
  %val1 = load volatile float, float addrspace(1)* %in.gep.1, align 4
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  store float %val0, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; GCN-LABEL: @simple_write2_two_val_f32_volatile_0
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_write2_b32
; GCN: ds_write_b32 {{v[0-9]+}}, {{v[0-9]+}}
; GCN: ds_write_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_two_val_f32_volatile_0(float addrspace(1)* %C, float addrspace(1)* %in0, float addrspace(1)* %in1) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in0.gep = getelementptr float, float addrspace(1)* %in0, i32 %x.i
  %in1.gep = getelementptr float, float addrspace(1)* %in1, i32 %x.i
  %val0 = load volatile float, float addrspace(1)* %in0.gep, align 4
  %val1 = load volatile float, float addrspace(1)* %in1.gep, align 4
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  store volatile float %val0, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; GCN-LABEL: @simple_write2_two_val_f32_volatile_1
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_write2_b32
; GCN: ds_write_b32 {{v[0-9]+}}, {{v[0-9]+}}
; GCN: ds_write_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_two_val_f32_volatile_1(float addrspace(1)* %C, float addrspace(1)* %in0, float addrspace(1)* %in1) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in0.gep = getelementptr float, float addrspace(1)* %in0, i32 %x.i
  %in1.gep = getelementptr float, float addrspace(1)* %in1, i32 %x.i
  %val0 = load volatile float, float addrspace(1)* %in0.gep, align 4
  %val1 = load volatile float, float addrspace(1)* %in1.gep, align 4
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  store float %val0, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  store volatile float %val1, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; 2 data subregisters from different super registers.
; GCN-LABEL: {{^}}simple_write2_two_val_subreg2_mixed_f32:
; GFX9-NOT: m0

; CI: buffer_load_dwordx2 v{{\[}}[[VAL0:[0-9]+]]:{{[0-9]+\]}}
; CI: buffer_load_dwordx2 v{{\[[0-9]+}}:[[VAL1:[0-9]+]]{{\]}}
; CI-DAG: s_mov_b32 m0

; GCN-DAG: v_lshlrev_b32_e32 [[VPTR:v[0-9]+]], 2, v{{[0-9]+}}

; GFX9: global_load_dwordx2 v{{\[}}[[VAL0:[0-9]+]]:{{[0-9]+\]}}
; GFX9: global_load_dwordx2 v{{\[[0-9]+}}:[[VAL1:[0-9]+]]{{\]}}

; GCN: ds_write2_b32 [[VPTR]], v[[VAL0]], v[[VAL1]] offset1:8
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_two_val_subreg2_mixed_f32(float addrspace(1)* %C, <2 x float> addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in.gep.0 = getelementptr <2 x float>, <2 x float> addrspace(1)* %in, i32 %x.i
  %in.gep.1 = getelementptr <2 x float>, <2 x float> addrspace(1)* %in.gep.0, i32 1
  %val0 = load volatile <2 x float>, <2 x float> addrspace(1)* %in.gep.0, align 8
  %val1 = load volatile <2 x float>, <2 x float> addrspace(1)* %in.gep.1, align 8
  %val0.0 = extractelement <2 x float> %val0, i32 0
  %val1.1 = extractelement <2 x float> %val1, i32 1
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  store float %val0.0, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  store float %val1.1, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; GCN-LABEL: @simple_write2_two_val_subreg2_f32
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: {{buffer|global}}_load_dwordx2 v{{\[}}[[VAL0:[0-9]+]]:[[VAL1:[0-9]+]]{{\]}}
; GCN-DAG: v_lshlrev_b32_e32 [[VPTR:v[0-9]+]], 2, v{{[0-9]+}}
; GCN: ds_write2_b32 [[VPTR]], v[[VAL0]], v[[VAL1]] offset1:8
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_two_val_subreg2_f32(float addrspace(1)* %C, <2 x float> addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in.gep = getelementptr <2 x float>, <2 x float> addrspace(1)* %in, i32 %x.i
  %val = load <2 x float>, <2 x float> addrspace(1)* %in.gep, align 8
  %val0 = extractelement <2 x float> %val, i32 0
  %val1 = extractelement <2 x float> %val, i32 1
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  store float %val0, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; GCN-LABEL: @simple_write2_two_val_subreg4_f32
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: {{buffer|global}}_load_dwordx4 v{{\[}}[[VAL0:[0-9]+]]:[[VAL1:[0-9]+]]{{\]}}
; GCN-DAG: v_lshlrev_b32_e32 [[VPTR:v[0-9]+]], 2, v{{[0-9]+}}
; GCN: ds_write2_b32 [[VPTR]], v[[VAL0]], v[[VAL1]] offset1:8
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_two_val_subreg4_f32(float addrspace(1)* %C, <4 x float> addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in.gep = getelementptr <4 x float>, <4 x float> addrspace(1)* %in, i32 %x.i
  %val = load <4 x float>, <4 x float> addrspace(1)* %in.gep, align 16
  %val0 = extractelement <4 x float> %val, i32 0
  %val1 = extractelement <4 x float> %val, i32 3
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  store float %val0, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; GCN-LABEL: @simple_write2_two_val_max_offset_f32
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; CI-DAG: buffer_load_dword [[VAL0:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CI-DAG: buffer_load_dword [[VAL1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4

; GFX9-DAG: global_load_dword [[VAL0:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, off{{$}}
; GFX9-DAG: global_load_dword [[VAL1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, off offset:4

; GCN-DAG: v_lshlrev_b32_e32 [[VPTR:v[0-9]+]], 2, v{{[0-9]+}}
; GCN: ds_write2_b32 [[VPTR]], [[VAL0]], [[VAL1]] offset1:255
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_two_val_max_offset_f32(float addrspace(1)* %C, float addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in.gep.0 = getelementptr float, float addrspace(1)* %in, i32 %x.i
  %in.gep.1 = getelementptr float, float addrspace(1)* %in.gep.0, i32 1
  %val0 = load volatile float, float addrspace(1)* %in.gep.0, align 4
  %val1 = load volatile float, float addrspace(1)* %in.gep.1, align 4
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  store float %val0, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 255
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; GCN-LABEL: @simple_write2_two_val_too_far_f32
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:1028
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_two_val_too_far_f32(float addrspace(1)* %C, float addrspace(1)* %in0, float addrspace(1)* %in1) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in0.gep = getelementptr float, float addrspace(1)* %in0, i32 %x.i
  %in1.gep = getelementptr float, float addrspace(1)* %in1, i32 %x.i
  %val0 = load float, float addrspace(1)* %in0.gep, align 4
  %val1 = load float, float addrspace(1)* %in1.gep, align 4
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  store float %val0, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 257
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  store float %val1, float addrspace(3)* %arrayidx1, align 4
  ret void
}

; GCN-LABEL: @simple_write2_two_val_f32_x2
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_write2_b32 [[BASEADDR:v[0-9]+]], [[VAL0:v[0-9]+]], [[VAL1:v[0-9]+]] offset1:8
; GCN: ds_write2_b32 [[BASEADDR:v[0-9]+]], [[VAL0]], [[VAL1]] offset0:11 offset1:27
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_two_val_f32_x2(float addrspace(1)* %C, float addrspace(1)* %in0, float addrspace(1)* %in1) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in0.gep = getelementptr float, float addrspace(1)* %in0, i32 %tid.x
  %in1.gep = getelementptr float, float addrspace(1)* %in1, i32 %tid.x
  %val0 = load float, float addrspace(1)* %in0.gep, align 4
  %val1 = load float, float addrspace(1)* %in1.gep, align 4

  %idx.0 = add nsw i32 %tid.x, 0
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.0
  store float %val0, float addrspace(3)* %arrayidx0, align 4

  %idx.1 = add nsw i32 %tid.x, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.1
  store float %val1, float addrspace(3)* %arrayidx1, align 4

  %idx.2 = add nsw i32 %tid.x, 11
  %arrayidx2 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.2
  store float %val0, float addrspace(3)* %arrayidx2, align 4

  %idx.3 = add nsw i32 %tid.x, 27
  %arrayidx3 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.3
  store float %val1, float addrspace(3)* %arrayidx3, align 4

  ret void
}

; GCN-LABEL: @simple_write2_two_val_f32_x2_nonzero_base
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_write2_b32 [[BASEADDR:v[0-9]+]], [[VAL0:v[0-9]+]], [[VAL1:v[0-9]+]] offset0:3 offset1:8
; GCN: ds_write2_b32 [[BASEADDR:v[0-9]+]], [[VAL0]], [[VAL1]] offset0:11 offset1:27
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_two_val_f32_x2_nonzero_base(float addrspace(1)* %C, float addrspace(1)* %in0, float addrspace(1)* %in1) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in0.gep = getelementptr float, float addrspace(1)* %in0, i32 %tid.x
  %in1.gep = getelementptr float, float addrspace(1)* %in1, i32 %tid.x
  %val0 = load float, float addrspace(1)* %in0.gep, align 4
  %val1 = load float, float addrspace(1)* %in1.gep, align 4

  %idx.0 = add nsw i32 %tid.x, 3
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.0
  store float %val0, float addrspace(3)* %arrayidx0, align 4

  %idx.1 = add nsw i32 %tid.x, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.1
  store float %val1, float addrspace(3)* %arrayidx1, align 4

  %idx.2 = add nsw i32 %tid.x, 11
  %arrayidx2 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.2
  store float %val0, float addrspace(3)* %arrayidx2, align 4

  %idx.3 = add nsw i32 %tid.x, 27
  %arrayidx3 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.3
  store float %val1, float addrspace(3)* %arrayidx3, align 4

  ret void
}

; GCN-LABEL: @write2_ptr_subreg_arg_two_val_f32
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_write2_b32
; GCN: ds_write_b32
; GCN: ds_write_b32
; GCN: s_endpgm
define amdgpu_kernel void @write2_ptr_subreg_arg_two_val_f32(float addrspace(1)* %C, float addrspace(1)* %in0, float addrspace(1)* %in1, <2 x float addrspace(3)*> %lds.ptr) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in0.gep = getelementptr float, float addrspace(1)* %in0, i32 %x.i
  %in1.gep = getelementptr float, float addrspace(1)* %in1, i32 %x.i
  %val0 = load float, float addrspace(1)* %in0.gep, align 4
  %val1 = load float, float addrspace(1)* %in1.gep, align 4

  %index.0 = insertelement <2 x i32> undef, i32 %x.i, i32 0
  %index.1 = insertelement <2 x i32> %index.0, i32 8, i32 0
  %gep = getelementptr inbounds float, <2 x float addrspace(3)*> %lds.ptr, <2 x i32> %index.1
  %gep.0 = extractelement <2 x float addrspace(3)*> %gep, i32 0
  %gep.1 = extractelement <2 x float addrspace(3)*> %gep, i32 1

  ; Apply an additional offset after the vector that will be more obviously folded.
  %gep.1.offset = getelementptr float, float addrspace(3)* %gep.1, i32 8
  store float %val0, float addrspace(3)* %gep.0, align 4

  %add.x = add nsw i32 %x.i, 8
  store float %val1, float addrspace(3)* %gep.1.offset, align 4
  ret void
}

; GCN-LABEL: @simple_write2_one_val_f64
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: {{buffer|global}}_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]],
; GCN-DAG: v_lshlrev_b32_e32 [[VPTR:v[0-9]+]], 3, v{{[0-9]+}}
; GCN: ds_write2_b64 [[VPTR]], [[VAL]], [[VAL]] offset1:8
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_one_val_f64(double addrspace(1)* %C, double addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in.gep = getelementptr double, double addrspace(1)* %in, i32 %x.i
  %val = load double, double addrspace(1)* %in.gep, align 8
  %arrayidx0 = getelementptr inbounds [512 x double], [512 x double] addrspace(3)* @lds.f64, i32 0, i32 %x.i
  store double %val, double addrspace(3)* %arrayidx0, align 8
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x double], [512 x double] addrspace(3)* @lds.f64, i32 0, i32 %add.x
  store double %val, double addrspace(3)* %arrayidx1, align 8
  ret void
}

; GCN-LABEL: @misaligned_simple_write2_one_val_f64
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: {{buffer|global}}_load_dwordx2 v{{\[}}[[VAL0:[0-9]+]]:[[VAL1:[0-9]+]]{{\]}}
; GCN-DAG: v_lshlrev_b32_e32 [[VPTR:v[0-9]+]], 3, v{{[0-9]+}}
; GCN: ds_write2_b32 [[VPTR]], v[[VAL0]], v[[VAL1]] offset1:1
; GCN: ds_write2_b32 [[VPTR]], v[[VAL0]], v[[VAL1]] offset0:14 offset1:15
; GCN: s_endpgm
define amdgpu_kernel void @misaligned_simple_write2_one_val_f64(double addrspace(1)* %C, double addrspace(1)* %in, double addrspace(3)* %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in.gep = getelementptr double, double addrspace(1)* %in, i32 %x.i
  %val = load double, double addrspace(1)* %in.gep, align 8
  %arrayidx0 = getelementptr inbounds double, double addrspace(3)* %lds, i32 %x.i
  store double %val, double addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 7
  %arrayidx1 = getelementptr inbounds double, double addrspace(3)* %lds, i32 %add.x
  store double %val, double addrspace(3)* %arrayidx1, align 4
  ret void
}

; GCN-LABEL: @simple_write2_two_val_f64
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; CI-DAG: buffer_load_dwordx2 [[VAL0:v\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CI-DAG: buffer_load_dwordx2 [[VAL1:v\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8

; GFX9-DAG: global_load_dwordx2 [[VAL0:v\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, off{{$}}
; GFX9-DAG: global_load_dwordx2 [[VAL1:v\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, off offset:8


; GCN-DAG: v_lshlrev_b32_e32 [[VPTR:v[0-9]+]], 3, v{{[0-9]+}}
; GCN: ds_write2_b64 [[VPTR]], [[VAL0]], [[VAL1]] offset1:8
; GCN: s_endpgm
define amdgpu_kernel void @simple_write2_two_val_f64(double addrspace(1)* %C, double addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in.gep.0 = getelementptr double, double addrspace(1)* %in, i32 %x.i
  %in.gep.1 = getelementptr double, double addrspace(1)* %in.gep.0, i32 1
  %val0 = load volatile double, double addrspace(1)* %in.gep.0, align 8
  %val1 = load volatile double, double addrspace(1)* %in.gep.1, align 8
  %arrayidx0 = getelementptr inbounds [512 x double], [512 x double] addrspace(3)* @lds.f64, i32 0, i32 %x.i
  store double %val0, double addrspace(3)* %arrayidx0, align 8
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x double], [512 x double] addrspace(3)* @lds.f64, i32 0, i32 %add.x
  store double %val1, double addrspace(3)* %arrayidx1, align 8
  ret void
}

@foo = addrspace(3) global [4 x i32] undef, align 4

; GCN-LABEL: @store_constant_adjacent_offsets
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN: ds_write2_b32 [[ZERO]], v{{[0-9]+}}, v{{[0-9]+}} offset1:1
define amdgpu_kernel void @store_constant_adjacent_offsets() {
  store i32 123, i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @foo, i32 0, i32 0), align 4
  store i32 123, i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @foo, i32 0, i32 1), align 4
  ret void
}

; GCN-LABEL: @store_constant_disjoint_offsets
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[VAL:v[0-9]+]], 0x7b{{$}}
; GCN-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN: ds_write2_b32 [[ZERO]], [[VAL]], [[VAL]] offset1:2
define amdgpu_kernel void @store_constant_disjoint_offsets() {
  store i32 123, i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @foo, i32 0, i32 0), align 4
  store i32 123, i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @foo, i32 0, i32 2), align 4
  ret void
}

@bar = addrspace(3) global [4 x i64] undef, align 4

; GCN-LABEL: @store_misaligned64_constant_offsets
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN-DAG: ds_write2_b32 [[ZERO]], v{{[0-9]+}}, v{{[0-9]+}} offset1:1
; GCN-DAG: ds_write2_b32 [[ZERO]], v{{[0-9]+}}, v{{[0-9]+}} offset0:2 offset1:3
; GCN: s_endpgm
define amdgpu_kernel void @store_misaligned64_constant_offsets() {
  store i64 123, i64 addrspace(3)* getelementptr inbounds ([4 x i64], [4 x i64] addrspace(3)* @bar, i32 0, i32 0), align 4
  store i64 123, i64 addrspace(3)* getelementptr inbounds ([4 x i64], [4 x i64] addrspace(3)* @bar, i32 0, i32 1), align 4
  ret void
}

@bar.large = addrspace(3) global [4096 x i64] undef, align 4

; GCN-LABEL: @store_misaligned64_constant_large_offsets
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[BASE0:v[0-9]+]], 0x7ff8{{$}}
; GCN-DAG: v_mov_b32_e32 [[BASE1:v[0-9]+]], 0x4000{{$}}
; GCN-DAG: ds_write2_b32 [[BASE0]], v{{[0-9]+}}, v{{[0-9]+}} offset1:1
; GCN-DAG: ds_write2_b32 [[BASE1]], v{{[0-9]+}}, v{{[0-9]+}} offset1:1
; GCN: s_endpgm
define amdgpu_kernel void @store_misaligned64_constant_large_offsets() {
  store i64 123, i64 addrspace(3)* getelementptr inbounds ([4096 x i64], [4096 x i64] addrspace(3)* @bar.large, i32 0, i32 2048), align 4
  store i64 123, i64 addrspace(3)* getelementptr inbounds ([4096 x i64], [4096 x i64] addrspace(3)* @bar.large, i32 0, i32 4095), align 4
  ret void
}

@sgemm.lA = internal unnamed_addr addrspace(3) global [264 x float] undef, align 4
@sgemm.lB = internal unnamed_addr addrspace(3) global [776 x float] undef, align 4

define amdgpu_kernel void @write2_sgemm_sequence(float addrspace(1)* %C, i32 %lda, i32 %ldb, float addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workgroup.id.x() #1
  %y.i = tail call i32 @llvm.amdgcn.workitem.id.y() #1
  %val = load float, float addrspace(1)* %in
  %arrayidx44 = getelementptr inbounds [264 x float], [264 x float] addrspace(3)* @sgemm.lA, i32 0, i32 %x.i
  store float %val, float addrspace(3)* %arrayidx44, align 4
  %add47 = add nsw i32 %x.i, 1
  %arrayidx48 = getelementptr inbounds [264 x float], [264 x float] addrspace(3)* @sgemm.lA, i32 0, i32 %add47
  store float %val, float addrspace(3)* %arrayidx48, align 4
  %add51 = add nsw i32 %x.i, 16
  %arrayidx52 = getelementptr inbounds [264 x float], [264 x float] addrspace(3)* @sgemm.lA, i32 0, i32 %add51
  store float %val, float addrspace(3)* %arrayidx52, align 4
  %add55 = add nsw i32 %x.i, 17
  %arrayidx56 = getelementptr inbounds [264 x float], [264 x float] addrspace(3)* @sgemm.lA, i32 0, i32 %add55
  store float %val, float addrspace(3)* %arrayidx56, align 4
  %arrayidx60 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %y.i
  store float %val, float addrspace(3)* %arrayidx60, align 4
  %add63 = add nsw i32 %y.i, 1
  %arrayidx64 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %add63
  store float %val, float addrspace(3)* %arrayidx64, align 4
  %add67 = add nsw i32 %y.i, 32
  %arrayidx68 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %add67
  store float %val, float addrspace(3)* %arrayidx68, align 4
  %add71 = add nsw i32 %y.i, 33
  %arrayidx72 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %add71
  store float %val, float addrspace(3)* %arrayidx72, align 4
  %add75 = add nsw i32 %y.i, 64
  %arrayidx76 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %add75
  store float %val, float addrspace(3)* %arrayidx76, align 4
  %add79 = add nsw i32 %y.i, 65
  %arrayidx80 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %add79
  store float %val, float addrspace(3)* %arrayidx80, align 4
  ret void
}

; GCN-LABEL: {{^}}simple_write2_v4f32_superreg_align4:
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_write2_b32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} offset0:2 offset1:3{{$}}
; GCN: ds_write2_b32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}} offset1:1{{$}}
define amdgpu_kernel void @simple_write2_v4f32_superreg_align4(<4 x float> addrspace(3)* %out, <4 x float> addrspace(1)* %in) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %in.gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in
  %val0 = load <4 x float>, <4 x float> addrspace(1)* %in.gep, align 4
  %out.gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(3)* %out, i32 %x.i
  store <4 x float> %val0, <4 x float> addrspace(3)* %out.gep, align 4
  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.x() #1
declare i32 @llvm.amdgcn.workgroup.id.y() #1
declare i32 @llvm.amdgcn.workitem.id.x() #1
declare i32 @llvm.amdgcn.workitem.id.y() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { convergent nounwind }
