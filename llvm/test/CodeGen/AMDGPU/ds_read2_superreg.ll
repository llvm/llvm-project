; RUN: llc -mtriple=amdgcn -mcpu=bonaire -verify-machineinstrs -mattr=+load-store-opt,-enable-ds128 < %s | FileCheck --check-prefix=CI %s

@lds = addrspace(3) global [512 x float] undef, align 4
@lds.v2 = addrspace(3) global [512 x <2 x float>] undef, align 4
@lds.v3 = addrspace(3) global [512 x <3 x float>] undef, align 4
@lds.v4 = addrspace(3) global [512 x <4 x float>] undef, align 4
@lds.v8 = addrspace(3) global [512 x <8 x float>] undef, align 4
@lds.v16 = addrspace(3) global [512 x <16 x float>] undef, align 4

; CI-LABEL: {{^}}simple_read2_v2f32_superreg_align4:
; CI: ds_read2_b32 [[RESULT:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}} offset1:1{{$}}
; CI: s_waitcnt lgkmcnt(0)
; CI: buffer_store_dwordx2 [[RESULT]]
; CI: s_endpgm
define amdgpu_kernel void @simple_read2_v2f32_superreg_align4(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds  [512 x <2 x float>], ptr addrspace(3) @lds.v2, i32 0, i32 %x.i
  %val0 = load <2 x float>, ptr addrspace(3) %arrayidx0, align 4
  %out.gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %out, i32 %x.i
  store <2 x float> %val0, ptr addrspace(1) %out.gep
  ret void
}

; CI-LABEL: {{^}}simple_read2_v2f32_superreg:
; CI: ds_read_b64 [[RESULT:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}{{$}}
; CI: s_waitcnt lgkmcnt(0)
; CI: buffer_store_dwordx2 [[RESULT]]
; CI: s_endpgm
define amdgpu_kernel void @simple_read2_v2f32_superreg(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <2 x float>], ptr addrspace(3) @lds.v2, i32 0, i32 %x.i
  %val0 = load <2 x float>, ptr addrspace(3) %arrayidx0
  %out.gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %out, i32 %x.i
  store <2 x float> %val0, ptr addrspace(1) %out.gep
  ret void
}

; CI-LABEL: {{^}}simple_read2_v4f32_superreg_align4:
; CI-DAG: ds_read2_b32 v[[[REG_X:[0-9]+]]:[[REG_Y:[0-9]+]]], v{{[0-9]+}} offset1:1{{$}}
; CI-DAG: ds_read2_b32 v[[[REG_Z:[0-9]+]]:[[REG_W:[0-9]+]]], v{{[0-9]+}} offset0:2 offset1:3{{$}}
; CI-DAG: v_add_f32_e32 v[[ADD0:[0-9]+]], v[[REG_X]], v[[REG_Z]]
; CI-DAG: v_add_f32_e32 v[[ADD1:[0-9]+]], v[[REG_Y]], v[[REG_W]]
; CI: v_add_f32_e32 v[[ADD2:[0-9]+]], v[[ADD0]], v[[ADD1]]
; CI: buffer_store_dword v[[ADD2]]
; CI: s_endpgm
define amdgpu_kernel void @simple_read2_v4f32_superreg_align4(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <4 x float>], ptr addrspace(3) @lds.v4, i32 0, i32 %x.i
  %val0 = load <4 x float>, ptr addrspace(3) %arrayidx0, align 4
  %elt0 = extractelement <4 x float> %val0, i32 0
  %elt1 = extractelement <4 x float> %val0, i32 1
  %elt2 = extractelement <4 x float> %val0, i32 2
  %elt3 = extractelement <4 x float> %val0, i32 3

  %add0 = fadd float %elt0, %elt2
  %add1 = fadd float %elt1, %elt3
  %add2 = fadd float %add0, %add1

  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i32 %x.i
  store float %add2, ptr addrspace(1) %out.gep
  ret void
}

; CI-LABEL: {{^}}simple_read2_v3f32_superreg_align4:
; CI-DAG: ds_read2_b32 v[[[REG_X:[0-9]+]]:[[REG_Y:[0-9]+]]], v{{[0-9]+}} offset1:1{{$}}
; CI-DAG: ds_read_b32 v[[REG_Z:[0-9]+]], v{{[0-9]+}} offset:8{{$}}
; CI-DAG: v_add_f32_e32 v[[ADD0:[0-9]+]], v[[REG_X]], v[[REG_Z]]
; CI-DAG: v_add_f32_e32 v[[ADD1:[0-9]+]], v[[ADD0]], v[[REG_Y]]
; CI: buffer_store_dword v[[ADD1]]
; CI: s_endpgm
define amdgpu_kernel void @simple_read2_v3f32_superreg_align4(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <3 x float>], ptr addrspace(3) @lds.v3, i32 0, i32 %x.i
  %val0 = load <3 x float>, ptr addrspace(3) %arrayidx0, align 4
  %elt0 = extractelement <3 x float> %val0, i32 0
  %elt1 = extractelement <3 x float> %val0, i32 1
  %elt2 = extractelement <3 x float> %val0, i32 2

  %add0 = fadd float %elt0, %elt2
  %add1 = fadd float %add0, %elt1

  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i32 %x.i
  store float %add1, ptr addrspace(1) %out.gep
  ret void
}

; CI-LABEL: {{^}}simple_read2_v4f32_superreg_align8:
; CI: ds_read2_b64 [[REG_ZW:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}} offset1:1{{$}}
; CI: buffer_store_dwordx4 [[REG_ZW]]
; CI: s_endpgm
define amdgpu_kernel void @simple_read2_v4f32_superreg_align8(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <4 x float>], ptr addrspace(3) @lds.v4, i32 0, i32 %x.i
  %val0 = load <4 x float>, ptr addrspace(3) %arrayidx0, align 8
  %out.gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %x.i
  store <4 x float> %val0, ptr addrspace(1) %out.gep
  ret void
}

; CI-LABEL: {{^}}simple_read2_v4f32_superreg:
; CI-DAG: ds_read2_b64 [[REG_ZW:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}} offset1:1{{$}}
; CI: buffer_store_dwordx4 [[REG_ZW]]
; CI: s_endpgm
define amdgpu_kernel void @simple_read2_v4f32_superreg(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <4 x float>], ptr addrspace(3) @lds.v4, i32 0, i32 %x.i
  %val0 = load <4 x float>, ptr addrspace(3) %arrayidx0
  %out.gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %x.i
  store <4 x float> %val0, ptr addrspace(1) %out.gep
  ret void
}

; FIXME: Extra moves shuffling superregister
; CI-LABEL: {{^}}simple_read2_v8f32_superreg:
; CI-DAG: ds_read2_b64 [[VEC_HI:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}} offset0:2 offset1:3{{$}}
; CI-DAG: ds_read2_b64 [[VEC_LO:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}} offset1:1{{$}}
; CI-DAG: buffer_store_dwordx4 [[VEC_HI]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64 offset:16
; CI-DAG: buffer_store_dwordx4 [[VEC_LO]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64{{$}}
; CI: s_endpgm
define amdgpu_kernel void @simple_read2_v8f32_superreg(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <8 x float>], ptr addrspace(3) @lds.v8, i32 0, i32 %x.i
  %val0 = load <8 x float>, ptr addrspace(3) %arrayidx0
  %out.gep = getelementptr inbounds <8 x float>, ptr addrspace(1) %out, i32 %x.i
  store <8 x float> %val0, ptr addrspace(1) %out.gep
  ret void
}

; FIXME: Extra moves shuffling superregister
; CI-LABEL: {{^}}simple_read2_v16f32_superreg:
; CI-DAG: ds_read2_b64 [[VEC0_3:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}} offset1:1{{$}}
; CI-DAG: ds_read2_b64 [[VEC4_7:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}} offset0:2 offset1:3{{$}}
; CI-DAG: ds_read2_b64 [[VEC8_11:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}} offset0:4 offset1:5{{$}}
; CI-DAG: ds_read2_b64 [[VEC12_15:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}} offset0:6 offset1:7{{$}}
; CI: s_waitcnt lgkmcnt(0)
; CI-DAG: buffer_store_dwordx4 [[VEC0_3]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64{{$}}
; CI-DAG: buffer_store_dwordx4 [[VEC4_7]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64 offset:16
; CI-DAG: buffer_store_dwordx4 [[VEC8_11]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64 offset:32
; CI-DAG: buffer_store_dwordx4 [[VEC12_15]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64 offset:48
; CI: s_endpgm
define amdgpu_kernel void @simple_read2_v16f32_superreg(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x <16 x float>], ptr addrspace(3) @lds.v16, i32 0, i32 %x.i
  %val0 = load <16 x float>, ptr addrspace(3) %arrayidx0
  %out.gep = getelementptr inbounds <16 x float>, ptr addrspace(1) %out, i32 %x.i
  store <16 x float> %val0, ptr addrspace(1) %out.gep
  ret void
}

; Do scalar loads into the super register we need.
; CI-LABEL: {{^}}simple_read2_v2f32_superreg_scalar_loads_align4:
; CI-DAG: ds_read2_b32 v[[[REG_ELT0:[0-9]+]]:[[REG_ELT1:[0-9]+]]], v{{[0-9]+}} offset1:1{{$}}
; CI-NOT: v_mov {{v[0-9]+}}, {{[sv][0-9]+}}
; CI: buffer_store_dwordx2 v[[[REG_ELT0]]:[[REG_ELT1]]]
; CI: s_endpgm
define amdgpu_kernel void @simple_read2_v2f32_superreg_scalar_loads_align4(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds, i32 0, i32 %x.i
  %arrayidx1 = getelementptr inbounds float, ptr addrspace(3) %arrayidx0, i32 1

  %val0 = load float, ptr addrspace(3) %arrayidx0
  %val1 = load float, ptr addrspace(3) %arrayidx1

  %vec.0 = insertelement <2 x float> undef, float %val0, i32 0
  %vec.1 = insertelement <2 x float> %vec.0, float %val1, i32 1

  %out.gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %out, i32 %x.i
  store <2 x float> %vec.1, ptr addrspace(1) %out.gep
  ret void
}

; Do scalar loads into the super register we need.
; CI-LABEL: {{^}}simple_read2_v4f32_superreg_scalar_loads_align4:
; CI-DAG: ds_read2_b32 v[[[REG_ELT0:[0-9]+]]:[[REG_ELT1:[0-9]+]]], v{{[0-9]+}} offset1:1{{$}}
; CI-DAG: ds_read2_b32 v[[[REG_ELT2:[0-9]+]]:[[REG_ELT3:[0-9]+]]], v{{[0-9]+}} offset0:2 offset1:3{{$}}
; CI-NOT: v_mov {{v[0-9]+}}, {{[sv][0-9]+}}
; CI: buffer_store_dwordx4 v[[[REG_ELT0]]:[[REG_ELT3]]]
; CI: s_endpgm
define amdgpu_kernel void @simple_read2_v4f32_superreg_scalar_loads_align4(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds, i32 0, i32 %x.i
  %arrayidx1 = getelementptr inbounds float, ptr addrspace(3) %arrayidx0, i32 1
  %arrayidx2 = getelementptr inbounds float, ptr addrspace(3) %arrayidx0, i32 2
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(3) %arrayidx0, i32 3

  %val0 = load float, ptr addrspace(3) %arrayidx0
  %val1 = load float, ptr addrspace(3) %arrayidx1
  %val2 = load float, ptr addrspace(3) %arrayidx2
  %val3 = load float, ptr addrspace(3) %arrayidx3

  %vec.0 = insertelement <4 x float> undef, float %val0, i32 0
  %vec.1 = insertelement <4 x float> %vec.0, float %val1, i32 1
  %vec.2 = insertelement <4 x float> %vec.1, float %val2, i32 2
  %vec.3 = insertelement <4 x float> %vec.2, float %val3, i32 3

  %out.gep = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %x.i
  store <4 x float> %vec.3, ptr addrspace(1) %out.gep
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.y() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { convergent nounwind }
