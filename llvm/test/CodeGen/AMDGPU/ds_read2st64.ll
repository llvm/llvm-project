; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -mattr=+load-store-opt < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -mattr=+load-store-opt < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

@lds = addrspace(3) global [512 x float] undef, align 4
@lds.f64 = addrspace(3) global [512 x double] undef, align 8


; GCN-LABEL: @simple_read2st64_f32_0_1
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2st64_b32 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]], v{{[0-9]+}} offset1:1
; GCN: s_waitcnt lgkmcnt(0)
; GCN: v_add_f32_e32 [[RESULT:v[0-9]+]], v[[LO_VREG]], v[[HI_VREG]]
; CI: buffer_store_dword [[RESULT]]
; GFX9: global_store_dword v{{[0-9]+}}, [[RESULT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @simple_read2st64_f32_0_1(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds, i32 0, i32 %x.i
  %val0 = load float, ptr addrspace(3) %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 64
  %arrayidx1 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds, i32 0, i32 %add.x
  %val1 = load float, ptr addrspace(3) %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i32 %x.i
  store float %sum, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: @simple_read2st64_f32_1_2
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2st64_b32 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]], v{{[0-9]+}} offset0:1 offset1:2
; GCN: s_waitcnt lgkmcnt(0)
; GCN: v_add_f32_e32 [[RESULT:v[0-9]+]], v[[LO_VREG]], v[[HI_VREG]]
; CI: buffer_store_dword [[RESULT]]
; GFX9: global_store_dword v{{[0-9]+}}, [[RESULT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @simple_read2st64_f32_1_2(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %add.x.0 = add nsw i32 %x.i, 64
  %arrayidx0 = getelementptr inbounds float, ptr addrspace(3) %lds, i32 %add.x.0
  %val0 = load float, ptr addrspace(3) %arrayidx0, align 4
  %add.x.1 = add nsw i32 %x.i, 128
  %arrayidx1 = getelementptr inbounds float, ptr addrspace(3) %lds, i32 %add.x.1
  %val1 = load float, ptr addrspace(3) %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i32 %x.i
  store float %sum, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: @simple_read2st64_f32_max_offset
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2st64_b32 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]], v{{[0-9]+}} offset0:1 offset1:255
; GCN: s_waitcnt lgkmcnt(0)
; GCN: v_add_f32_e32 [[RESULT:v[0-9]+]], v[[LO_VREG]], v[[HI_VREG]]
; CI: buffer_store_dword [[RESULT]]
; GFX9: global_store_dword v{{[0-9]+}}, [[RESULT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @simple_read2st64_f32_max_offset(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %add.x.0 = add nsw i32 %x.i, 64
  %arrayidx0 = getelementptr inbounds float, ptr addrspace(3) %lds, i32 %add.x.0
  %val0 = load float, ptr addrspace(3) %arrayidx0, align 4
  %add.x.1 = add nsw i32 %x.i, 16320
  %arrayidx1 = getelementptr inbounds float, ptr addrspace(3) %lds, i32 %add.x.1
  %val1 = load float, ptr addrspace(3) %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i32 %x.i
  store float %sum, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: @simple_read2st64_f32_over_max_offset
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2st64_b32
; GCN-DAG: v_add_{{i|u}}32_e32 [[BIGADD:v[0-9]+]], {{(vcc, )?}}0x10000, {{v[0-9]+}}
; GCN-DAG: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:256
; GCN-DAG: ds_read_b32 {{v[0-9]+}}, [[BIGADD]]{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @simple_read2st64_f32_over_max_offset(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %add.x.0 = add nsw i32 %x.i, 64
  %arrayidx0 = getelementptr inbounds float, ptr addrspace(3) %lds, i32 %add.x.0
  %val0 = load float, ptr addrspace(3) %arrayidx0, align 4
  %add.x.1 = add nsw i32 %x.i, 16384
  %arrayidx1 = getelementptr inbounds float, ptr addrspace(3) %lds, i32 %add.x.1
  %val1 = load float, ptr addrspace(3) %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i32 %x.i
  store float %sum, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: @odd_invalid_read2st64_f32_0
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2st64_b32
; GCN: s_endpgm
define amdgpu_kernel void @odd_invalid_read2st64_f32_0(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds, i32 0, i32 %x.i
  %val0 = load float, ptr addrspace(3) %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 63
  %arrayidx1 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds, i32 0, i32 %add.x
  %val1 = load float, ptr addrspace(3) %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i32 %x.i
  store float %sum, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: @odd_invalid_read2st64_f32_1
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2st64_b32
; GCN: s_endpgm
define amdgpu_kernel void @odd_invalid_read2st64_f32_1(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %add.x.0 = add nsw i32 %x.i, 64
  %arrayidx0 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds, i32 0, i32 %add.x.0
  %val0 = load float, ptr addrspace(3) %arrayidx0, align 4
  %add.x.1 = add nsw i32 %x.i, 127
  %arrayidx1 = getelementptr inbounds [512 x float], ptr addrspace(3) @lds, i32 0, i32 %add.x.1
  %val1 = load float, ptr addrspace(3) %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, ptr addrspace(1) %out, i32 %x.i
  store float %sum, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: @simple_read2st64_f64_0_1
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2st64_b64 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]], v{{[0-9]+}} offset1:1
; GCN: s_waitcnt lgkmcnt(0)
; GCN: v_add_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], v[[[LO_VREG]]:{{[0-9]+\]}}, v{{\[[0-9]+}}:[[HI_VREG]]]
; CI: buffer_store_dwordx2 [[RESULT]]
; GFX9: global_store_dwordx2 v{{[0-9]+}}, [[RESULT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @simple_read2st64_f64_0_1(ptr addrspace(1) %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x double], ptr addrspace(3) @lds.f64, i32 0, i32 %x.i
  %val0 = load double, ptr addrspace(3) %arrayidx0, align 8
  %add.x = add nsw i32 %x.i, 64
  %arrayidx1 = getelementptr inbounds [512 x double], ptr addrspace(3) @lds.f64, i32 0, i32 %add.x
  %val1 = load double, ptr addrspace(3) %arrayidx1, align 8
  %sum = fadd double %val0, %val1
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i32 %x.i
  store double %sum, ptr addrspace(1) %out.gep, align 8
  ret void
}

; GCN-LABEL: @simple_read2st64_f64_1_2
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2st64_b64 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]], v{{[0-9]+}} offset0:1 offset1:2
; GCN: s_waitcnt lgkmcnt(0)
; GCN: v_add_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], v[[[LO_VREG]]:{{[0-9]+\]}}, v{{\[[0-9]+}}:[[HI_VREG]]]

; CI: buffer_store_dwordx2 [[RESULT]]
; GFX9: global_store_dwordx2 v{{[0-9]+}}, [[RESULT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @simple_read2st64_f64_1_2(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %add.x.0 = add nsw i32 %x.i, 64
  %arrayidx0 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %add.x.0
  %val0 = load double, ptr addrspace(3) %arrayidx0, align 8
  %add.x.1 = add nsw i32 %x.i, 128
  %arrayidx1 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %add.x.1
  %val1 = load double, ptr addrspace(3) %arrayidx1, align 8
  %sum = fadd double %val0, %val1
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i32 %x.i
  store double %sum, ptr addrspace(1) %out.gep, align 8
  ret void
}

; Alignment only

; GCN-LABEL: @misaligned_read2st64_f64
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset1:1
; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset0:128 offset1:129
; GCN: s_endpgm
define amdgpu_kernel void @misaligned_read2st64_f64(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %x.i
  %val0 = load double, ptr addrspace(3) %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 64
  %arrayidx1 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %add.x
  %val1 = load double, ptr addrspace(3) %arrayidx1, align 4
  %sum = fadd double %val0, %val1
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i32 %x.i
  store double %sum, ptr addrspace(1) %out.gep, align 4
  ret void
}

; The maximum is not the usual 0xff because 0xff * 8 * 64 > 0xffff
; GCN-LABEL: @simple_read2st64_f64_max_offset
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2st64_b64 v[[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]], v{{[0-9]+}} offset0:4 offset1:127
; GCN: s_waitcnt lgkmcnt(0)
; GCN: v_add_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], v[[[LO_VREG]]:{{[0-9]+\]}}, v{{\[[0-9]+}}:[[HI_VREG]]]

; CI: buffer_store_dwordx2 [[RESULT]]
; GFX9: global_store_dwordx2 v{{[0-9]+}}, [[RESULT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @simple_read2st64_f64_max_offset(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %add.x.0 = add nsw i32 %x.i, 256
  %arrayidx0 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %add.x.0
  %val0 = load double, ptr addrspace(3) %arrayidx0, align 8
  %add.x.1 = add nsw i32 %x.i, 8128
  %arrayidx1 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %add.x.1
  %val1 = load double, ptr addrspace(3) %arrayidx1, align 8
  %sum = fadd double %val0, %val1
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i32 %x.i
  store double %sum, ptr addrspace(1) %out.gep, align 8
  ret void
}

; GCN-LABEL: @simple_read2st64_f64_over_max_offset
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2st64_b64
; GCN-DAG: ds_read_b64 {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset:512
; GCN-DAG: v_add_{{i|u}}32_e32 [[BIGADD:v[0-9]+]], {{(vcc, )?}}0x10000, {{v[0-9]+}}
; GCN: ds_read_b64 {{v\[[0-9]+:[0-9]+\]}}, [[BIGADD]]
; GCN: s_endpgm
define amdgpu_kernel void @simple_read2st64_f64_over_max_offset(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %add.x.0 = add nsw i32 %x.i, 64
  %arrayidx0 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %add.x.0
  %val0 = load double, ptr addrspace(3) %arrayidx0, align 8
  %add.x.1 = add nsw i32 %x.i, 8192
  %arrayidx1 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %add.x.1
  %val1 = load double, ptr addrspace(3) %arrayidx1, align 8
  %sum = fadd double %val0, %val1
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i32 %x.i
  store double %sum, ptr addrspace(1) %out.gep, align 8
  ret void
}

; GCN-LABEL: @invalid_read2st64_f64_odd_offset
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2st64_b64
; GCN: s_endpgm
define amdgpu_kernel void @invalid_read2st64_f64_odd_offset(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %add.x.0 = add nsw i32 %x.i, 64
  %arrayidx0 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %add.x.0
  %val0 = load double, ptr addrspace(3) %arrayidx0, align 8
  %add.x.1 = add nsw i32 %x.i, 8129
  %arrayidx1 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %add.x.1
  %val1 = load double, ptr addrspace(3) %arrayidx1, align 8
  %sum = fadd double %val0, %val1
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i32 %x.i
  store double %sum, ptr addrspace(1) %out.gep, align 8
  ret void
}

; The stride of 8 elements is 8 * 8 bytes. We need to make sure the
; stride in elements, not bytes, is a multiple of 64.

; GCN-LABEL: @byte_size_only_divisible_64_read2_f64
; CI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2st_b64
; GCN: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:8
; GCN: s_endpgm
define amdgpu_kernel void @byte_size_only_divisible_64_read2_f64(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %x.i
  %val0 = load double, ptr addrspace(3) %arrayidx0, align 8
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds double, ptr addrspace(3) %lds, i32 %add.x
  %val1 = load double, ptr addrspace(3) %arrayidx1, align 8
  %sum = fadd double %val0, %val1
  %out.gep = getelementptr inbounds double, ptr addrspace(1) %out, i32 %x.i
  store double %sum, ptr addrspace(1) %out.gep, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare i32 @llvm.amdgcn.workitem.id.y() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
