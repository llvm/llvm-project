; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -mattr=+load-store-opt < %s | FileCheck -enable-var-scope -strict-whitespace -check-prefixes=GCN,CI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -mattr=+load-store-opt,+flat-for-global,-unaligned-access-mode < %s | FileCheck -enable-var-scope -strict-whitespace -check-prefixes=GCN,GFX9,GFX9-ALIGNED %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -mattr=+load-store-opt,+flat-for-global,+unaligned-access-mode < %s | FileCheck -enable-var-scope -strict-whitespace -check-prefixes=GCN,GFX9,GFX9-UNALIGNED %s

; FIXME: We don't get cases where the address was an SGPR because we
; get a copy to the address register for each one.

@lds = addrspace(3) global [512 x float] undef, align 4
@lds.f64 = addrspace(3) global [512 x double] undef, align 8

; GCN-LABEL: {{^}}simple_read2_f32:
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b32 v{{\[}}[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]{{\]}}, v{{[0-9]+}} offset1:8
; GCN: s_waitcnt lgkmcnt(0)
; GCN: v_add_f32_e32 [[RESULT:v[0-9]+]], v[[LO_VREG]], v[[HI_VREG]]
; CI: buffer_store_dword [[RESULT]]
; GFX9: global_store_dword v{{[0-9]+}}, [[RESULT]], s{{\[[0-9]+:[0-9]+\]}}
; GCN: s_endpgm
define amdgpu_kernel void @simple_read2_f32(float addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  %val1 = load float, float addrspace(3)* %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}simple_read2_f32_max_offset:
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b32 v{{\[}}[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]{{\]}}, v{{[0-9]+}} offset1:255
; GCN: s_waitcnt lgkmcnt(0)
; GCN: v_add_f32_e32 [[RESULT:v[0-9]+]], v[[LO_VREG]], v[[HI_VREG]]

; CI: buffer_store_dword [[RESULT]]
; GFX9: global_store_dword v{{[0-9]+}}, [[RESULT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @simple_read2_f32_max_offset(float addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 255
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  %val1 = load float, float addrspace(3)* %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: @simple_read2_f32_too_far
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2_b32
; GCN: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:1028
; GCN: s_endpgm
define amdgpu_kernel void @simple_read2_f32_too_far(float addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 257
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  %val1 = load float, float addrspace(3)* %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: @simple_read2_f32_x2
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[BASEADDR:v[0-9]+]] offset1:8
; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[BASEADDR]] offset0:11 offset1:27
; GCN: s_endpgm
define amdgpu_kernel void @simple_read2_f32_x2(float addrspace(1)* %out) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 0
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.0
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4

  %idx.1 = add nsw i32 %tid.x, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.1
  %val1 = load float, float addrspace(3)* %arrayidx1, align 4
  %sum.0 = fadd float %val0, %val1

  %idx.2 = add nsw i32 %tid.x, 11
  %arrayidx2 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.2
  %val2 = load float, float addrspace(3)* %arrayidx2, align 4

  %idx.3 = add nsw i32 %tid.x, 27
  %arrayidx3 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.3
  %val3 = load float, float addrspace(3)* %arrayidx3, align 4
  %sum.1 = fadd float %val2, %val3

  %sum = fadd float %sum.0, %sum.1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %idx.0
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; Make sure there is an instruction between the two sets of reads.
; GCN-LABEL: @simple_read2_f32_x2_barrier
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[BASEADDR:v[0-9]+]] offset1:8
; GCN: s_barrier
; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[BASEADDR]] offset0:11 offset1:27
; GCN: s_endpgm
define amdgpu_kernel void @simple_read2_f32_x2_barrier(float addrspace(1)* %out) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 0
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.0
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4

  %idx.1 = add nsw i32 %tid.x, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.1
  %val1 = load float, float addrspace(3)* %arrayidx1, align 4
  %sum.0 = fadd float %val0, %val1

  call void @llvm.amdgcn.s.barrier() #2

  %idx.2 = add nsw i32 %tid.x, 11
  %arrayidx2 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.2
  %val2 = load float, float addrspace(3)* %arrayidx2, align 4

  %idx.3 = add nsw i32 %tid.x, 27
  %arrayidx3 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.3
  %val3 = load float, float addrspace(3)* %arrayidx3, align 4
  %sum.1 = fadd float %val2, %val3

  %sum = fadd float %sum.0, %sum.1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %idx.0
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; For some reason adding something to the base address for the first
; element results in only folding the inner pair.

; GCN-LABEL: @simple_read2_f32_x2_nonzero_base
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[BASEADDR:v[0-9]+]] offset0:2 offset1:8
; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[BASEADDR]] offset0:11 offset1:27
; GCN: s_endpgm
define amdgpu_kernel void @simple_read2_f32_x2_nonzero_base(float addrspace(1)* %out) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.0
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4

  %idx.1 = add nsw i32 %tid.x, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.1
  %val1 = load float, float addrspace(3)* %arrayidx1, align 4
  %sum.0 = fadd float %val0, %val1

  %idx.2 = add nsw i32 %tid.x, 11
  %arrayidx2 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.2
  %val2 = load float, float addrspace(3)* %arrayidx2, align 4

  %idx.3 = add nsw i32 %tid.x, 27
  %arrayidx3 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %idx.3
  %val3 = load float, float addrspace(3)* %arrayidx3, align 4
  %sum.1 = fadd float %val2, %val3

  %sum = fadd float %sum.0, %sum.1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %idx.0
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; Be careful of vectors of pointers. We don't know if the 2 pointers
; in the vectors are really the same base, so this is not safe to
; merge.
; Base pointers come from different subregister of same super
; register. We can't safely merge this.

; GCN-LABEL: @read2_ptr_is_subreg_arg_f32
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2_b32
; GCN: ds_read_b32
; GCN: ds_read_b32
; GCN: s_endpgm
define amdgpu_kernel void @read2_ptr_is_subreg_arg_f32(float addrspace(1)* %out, <2 x float addrspace(3)*> %lds.ptr) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %index.0 = insertelement <2 x i32> undef, i32 %x.i, i32 0
  %index.1 = insertelement <2 x i32> %index.0, i32 8, i32 0
  %gep = getelementptr inbounds float, <2 x float addrspace(3)*> %lds.ptr, <2 x i32> %index.1
  %gep.0 = extractelement <2 x float addrspace(3)*> %gep, i32 0
  %gep.1 = extractelement <2 x float addrspace(3)*> %gep, i32 1
  %val0 = load float, float addrspace(3)* %gep.0, align 4
  %val1 = load float, float addrspace(3)* %gep.1, align 4
  %add.x = add nsw i32 %x.i, 8
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; Apply a constant scalar offset after the pointer vector extract.  We
; are rejecting merges that have the same, constant 0 offset, so make
; sure we are really rejecting it because of the different
; subregisters.

; GCN-LABEL: @read2_ptr_is_subreg_arg_offset_f32
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2_b32
; GCN: ds_read_b32
; GCN: ds_read_b32
; GCN: s_endpgm
define amdgpu_kernel void @read2_ptr_is_subreg_arg_offset_f32(float addrspace(1)* %out, <2 x float addrspace(3)*> %lds.ptr) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %index.0 = insertelement <2 x i32> undef, i32 %x.i, i32 0
  %index.1 = insertelement <2 x i32> %index.0, i32 8, i32 0
  %gep = getelementptr inbounds float, <2 x float addrspace(3)*> %lds.ptr, <2 x i32> %index.1
  %gep.0 = extractelement <2 x float addrspace(3)*> %gep, i32 0
  %gep.1 = extractelement <2 x float addrspace(3)*> %gep, i32 1

  ; Apply an additional offset after the vector that will be more obviously folded.
  %gep.1.offset = getelementptr float, float addrspace(3)* %gep.1, i32 8

  %val0 = load float, float addrspace(3)* %gep.0, align 4
  %val1 = load float, float addrspace(3)* %gep.1.offset, align 4
  %add.x = add nsw i32 %x.i, 8
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}read2_ptr_is_subreg_f32:
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b32 {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset1:8{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @read2_ptr_is_subreg_f32(float addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %ptr.0 = insertelement <2 x [512 x float] addrspace(3)*> undef, [512 x float] addrspace(3)* @lds, i32 0
  %ptr.1 = insertelement <2 x [512 x float] addrspace(3)*> %ptr.0, [512 x float] addrspace(3)* @lds, i32 1
  %x.i.v.0 = insertelement <2 x i32> undef, i32 %x.i, i32 0
  %x.i.v.1 = insertelement <2 x i32> %x.i.v.0, i32 %x.i, i32 1
  %idx = add <2 x i32> %x.i.v.1, <i32 0, i32 8>
  %gep = getelementptr inbounds [512 x float], <2 x [512 x float] addrspace(3)*> %ptr.1, <2 x i32> <i32 0, i32 0>, <2 x i32> %idx
  %gep.0 = extractelement <2 x float addrspace(3)*> %gep, i32 0
  %gep.1 = extractelement <2 x float addrspace(3)*> %gep, i32 1
  %val0 = load float, float addrspace(3)* %gep.0, align 4
  %val1 = load float, float addrspace(3)* %gep.1, align 4
  %add.x = add nsw i32 %x.i, 8
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: @simple_read2_f32_volatile_0
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2_b32
; GCN: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @simple_read2_f32_volatile_0(float addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  %val0 = load volatile float, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  %val1 = load float, float addrspace(3)* %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: @simple_read2_f32_volatile_1
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2_b32
; GCN: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @simple_read2_f32_volatile_1(float addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %x.i
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %add.x
  %val1 = load volatile float, float addrspace(3)* %arrayidx1, align 4
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; Can't fold since not correctly aligned.
; XXX: This isn't really testing anything useful now. I think CI
; allows unaligned LDS accesses, which would be a problem here.
; GCN-LABEL: @unaligned_read2_f32
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; CI-COUNT-4: ds_read_u8
; GFX9-ALIGNED-4: ds_read_u8
; GFX9-UNALIGNED-4: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}}
; GCN: s_endpgm
define amdgpu_kernel void @unaligned_read2_f32(float addrspace(1)* %out, float addrspace(3)* %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds float, float addrspace(3)* %lds, i32 %x.i
  %val0 = load float, float addrspace(3)* %arrayidx0, align 1
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds float, float addrspace(3)* %lds, i32 %add.x
  %val1 = load float, float addrspace(3)* %arrayidx1, align 1
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: @misaligned_2_simple_read2_f32
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; CI-COUNT-2: ds_read_u16
; GFX9-ALIGNED-2: ds_read_u16
; GFX9-UNALIGNED-4: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}}
; GCN: s_endpgm
define amdgpu_kernel void @misaligned_2_simple_read2_f32(float addrspace(1)* %out, float addrspace(3)* %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds float, float addrspace(3)* %lds, i32 %x.i
  %val0 = load float, float addrspace(3)* %arrayidx0, align 2
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds float, float addrspace(3)* %lds, i32 %add.x
  %val1 = load float, float addrspace(3)* %arrayidx1, align 2
  %sum = fadd float %val0, %val1
  %out.gep = getelementptr inbounds float, float addrspace(1)* %out, i32 %x.i
  store float %sum, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: @simple_read2_f64
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_lshlrev_b32_e32 [[VOFS:v[0-9]+]], 3, {{v[0-9]+}}
; GCN-DAG: v_add_{{[iu]}}32_e32 [[VPTR:v[0-9]+]], {{(vcc, )?}}lds.f64@abs32@lo, [[VOFS]]
; GCN: ds_read2_b64 v{{\[}}[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]{{\]}}, [[VPTR]] offset1:8
; GCN: v_add_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], v{{\[}}[[LO_VREG]]:{{[0-9]+\]}}, v{{\[[0-9]+}}:[[HI_VREG]]{{\]}}

; CI: buffer_store_dwordx2 [[RESULT]]
; GFX9: global_store_dwordx2 v{{[0-9]+}}, [[RESULT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @simple_read2_f64(double addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x double], [512 x double] addrspace(3)* @lds.f64, i32 0, i32 %x.i
  %val0 = load double, double addrspace(3)* %arrayidx0, align 8
  %add.x = add nsw i32 %x.i, 8
  %arrayidx1 = getelementptr inbounds [512 x double], [512 x double] addrspace(3)* @lds.f64, i32 0, i32 %add.x
  %val1 = load double, double addrspace(3)* %arrayidx1, align 8
  %sum = fadd double %val0, %val1
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i32 %x.i
  store double %sum, double addrspace(1)* %out.gep, align 8
  ret void
}

; GCN-LABEL: @simple_read2_f64_max_offset
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b64 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:255
; GCN: s_endpgm
define amdgpu_kernel void @simple_read2_f64_max_offset(double addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x double], [512 x double] addrspace(3)* @lds.f64, i32 0, i32 %x.i
  %val0 = load double, double addrspace(3)* %arrayidx0, align 8
  %add.x = add nsw i32 %x.i, 255
  %arrayidx1 = getelementptr inbounds [512 x double], [512 x double] addrspace(3)* @lds.f64, i32 0, i32 %add.x
  %val1 = load double, double addrspace(3)* %arrayidx1, align 8
  %sum = fadd double %val0, %val1
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i32 %x.i
  store double %sum, double addrspace(1)* %out.gep, align 8
  ret void
}

; GCN-LABEL: @simple_read2_f64_too_far
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read2_b64
; GCN: ds_read_b64 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}
; GCN: ds_read_b64 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset:2056
; GCN: s_endpgm
define amdgpu_kernel void @simple_read2_f64_too_far(double addrspace(1)* %out) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds [512 x double], [512 x double] addrspace(3)* @lds.f64, i32 0, i32 %x.i
  %val0 = load double, double addrspace(3)* %arrayidx0, align 8
  %add.x = add nsw i32 %x.i, 257
  %arrayidx1 = getelementptr inbounds [512 x double], [512 x double] addrspace(3)* @lds.f64, i32 0, i32 %add.x
  %val1 = load double, double addrspace(3)* %arrayidx1, align 8
  %sum = fadd double %val0, %val1
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i32 %x.i
  store double %sum, double addrspace(1)* %out.gep, align 8
  ret void
}

; Alignment only 4
; GCN-LABEL: @misaligned_read2_f64
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset1:1
; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset0:14 offset1:15
; GCN: s_endpgm
define amdgpu_kernel void @misaligned_read2_f64(double addrspace(1)* %out, double addrspace(3)* %lds) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %arrayidx0 = getelementptr inbounds double, double addrspace(3)* %lds, i32 %x.i
  %val0 = load double, double addrspace(3)* %arrayidx0, align 4
  %add.x = add nsw i32 %x.i, 7
  %arrayidx1 = getelementptr inbounds double, double addrspace(3)* %lds, i32 %add.x
  %val1 = load double, double addrspace(3)* %arrayidx1, align 4
  %sum = fadd double %val0, %val1
  %out.gep = getelementptr inbounds double, double addrspace(1)* %out, i32 %x.i
  store double %sum, double addrspace(1)* %out.gep, align 4
  ret void
}

@foo = addrspace(3) global [4 x i32] undef, align 4

; GCN-LABEL: @load_constant_adjacent_offsets
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], foo@abs32@lo{{$}}
; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[PTR]] offset1:1
define amdgpu_kernel void @load_constant_adjacent_offsets(i32 addrspace(1)* %out) {
  %val0 = load i32, i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @foo, i32 0, i32 0), align 4
  %val1 = load i32, i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @foo, i32 0, i32 1), align 4
  %sum = add i32 %val0, %val1
  store i32 %sum, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: @load_constant_disjoint_offsets
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], foo@abs32@lo{{$}}
; GCN: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[PTR]] offset1:2
define amdgpu_kernel void @load_constant_disjoint_offsets(i32 addrspace(1)* %out) {
  %val0 = load i32, i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @foo, i32 0, i32 0), align 4
  %val1 = load i32, i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @foo, i32 0, i32 2), align 4
  %sum = add i32 %val0, %val1
  store i32 %sum, i32 addrspace(1)* %out, align 4
  ret void
}

@bar = addrspace(3) global [4 x i64] undef, align 4

; GCN-LABEL: @load_misaligned64_constant_offsets
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[PTR:v[0-9]+]], bar@abs32@lo{{$}}
; CI: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[PTR]] offset1:1
; CI: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[PTR]] offset0:2 offset1:3
; GFX9: ds_read_b128 v{{\[[0-9]+:[0-9]+\]}}, [[PTR]]
define amdgpu_kernel void @load_misaligned64_constant_offsets(i64 addrspace(1)* %out) {
  %val0 = load i64, i64 addrspace(3)* getelementptr inbounds ([4 x i64], [4 x i64] addrspace(3)* @bar, i32 0, i32 0), align 4
  %val1 = load i64, i64 addrspace(3)* getelementptr inbounds ([4 x i64], [4 x i64] addrspace(3)* @bar, i32 0, i32 1), align 4
  %sum = add i64 %val0, %val1
  store i64 %sum, i64 addrspace(1)* %out, align 8
  ret void
}

@bar.large = addrspace(3) global [4096 x i64] undef, align 4

; GCN-LABEL: @load_misaligned64_constant_large_offsets
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: s_mov_b32 [[SBASE0:s[0-9]+]], bar.large@abs32@lo
; GCN-DAG: s_add_i32 [[SBASE1:s[0-9]+]], [[SBASE0]], 0x4000{{$}}
; GCN-DAG: s_addk_i32 [[SBASE0]], 0x7ff8{{$}}
; GCN-DAG: v_mov_b32_e32 [[VBASE0:v[0-9]+]], [[SBASE0]]
; GCN-DAG: v_mov_b32_e32 [[VBASE1:v[0-9]+]], [[SBASE1]]
; GCN-DAG: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[VBASE0]] offset1:1
; GCN-DAG: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[VBASE1]] offset1:1
; GCN: s_endpgm
define amdgpu_kernel void @load_misaligned64_constant_large_offsets(i64 addrspace(1)* %out) {
  %val0 = load i64, i64 addrspace(3)* getelementptr inbounds ([4096 x i64], [4096 x i64] addrspace(3)* @bar.large, i32 0, i32 2048), align 4
  %val1 = load i64, i64 addrspace(3)* getelementptr inbounds ([4096 x i64], [4096 x i64] addrspace(3)* @bar.large, i32 0, i32 4095), align 4
  %sum = add i64 %val0, %val1
  store i64 %sum, i64 addrspace(1)* %out, align 8
  ret void
}

@sgemm.lA = internal unnamed_addr addrspace(3) global [264 x float] undef, align 4
@sgemm.lB = internal unnamed_addr addrspace(3) global [776 x float] undef, align 4

; GCN-LABEL: {{^}}sgemm_inner_loop_read2_sequence:
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

define amdgpu_kernel void @sgemm_inner_loop_read2_sequence(float addrspace(1)* %C, i32 %lda, i32 %ldb) #0 {
  %x.i = tail call i32 @llvm.amdgcn.workgroup.id.x() #1
  %y.i = tail call i32 @llvm.amdgcn.workitem.id.y() #1
  %arrayidx44 = getelementptr inbounds [264 x float], [264 x float] addrspace(3)* @sgemm.lA, i32 0, i32 %x.i
  %tmp16 = load float, float addrspace(3)* %arrayidx44, align 4
  %add47 = add nsw i32 %x.i, 1
  %arrayidx48 = getelementptr inbounds [264 x float], [264 x float] addrspace(3)* @sgemm.lA, i32 0, i32 %add47
  %tmp17 = load float, float addrspace(3)* %arrayidx48, align 4
  %add51 = add nsw i32 %x.i, 16
  %arrayidx52 = getelementptr inbounds [264 x float], [264 x float] addrspace(3)* @sgemm.lA, i32 0, i32 %add51
  %tmp18 = load float, float addrspace(3)* %arrayidx52, align 4
  %add55 = add nsw i32 %x.i, 17
  %arrayidx56 = getelementptr inbounds [264 x float], [264 x float] addrspace(3)* @sgemm.lA, i32 0, i32 %add55
  %tmp19 = load float, float addrspace(3)* %arrayidx56, align 4
  %arrayidx60 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %y.i
  %tmp20 = load float, float addrspace(3)* %arrayidx60, align 4
  %add63 = add nsw i32 %y.i, 1
  %arrayidx64 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %add63
  %tmp21 = load float, float addrspace(3)* %arrayidx64, align 4
  %add67 = add nsw i32 %y.i, 32
  %arrayidx68 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %add67
  %tmp22 = load float, float addrspace(3)* %arrayidx68, align 4
  %add71 = add nsw i32 %y.i, 33
  %arrayidx72 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %add71
  %tmp23 = load float, float addrspace(3)* %arrayidx72, align 4
  %add75 = add nsw i32 %y.i, 64
  %arrayidx76 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %add75
  %tmp24 = load float, float addrspace(3)* %arrayidx76, align 4
  %add79 = add nsw i32 %y.i, 65
  %arrayidx80 = getelementptr inbounds [776 x float], [776 x float] addrspace(3)* @sgemm.lB, i32 0, i32 %add79
  %tmp25 = load float, float addrspace(3)* %arrayidx80, align 4
  %sum.0 = fadd float %tmp16, %tmp17
  %sum.1 = fadd float %sum.0, %tmp18
  %sum.2 = fadd float %sum.1, %tmp19
  %sum.3 = fadd float %sum.2, %tmp20
  %sum.4 = fadd float %sum.3, %tmp21
  %sum.5 = fadd float %sum.4, %tmp22
  %sum.6 = fadd float %sum.5, %tmp23
  %sum.7 = fadd float %sum.6, %tmp24
  %sum.8 = fadd float %sum.7, %tmp25
  store float %sum.8, float addrspace(1)* %C, align 4
  ret void
}

; GCN-LABEL: {{^}}misaligned_read2_v2i32:
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @misaligned_read2_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(3)* %in) #0 {
  %load = load <2 x i32>, <2 x i32> addrspace(3)* %in, align 4
  store <2 x i32> %load, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}misaligned_read2_i64:
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @misaligned_read2_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %in) #0 {
  %load = load i64, i64 addrspace(3)* %in, align 4
  store i64 %load, i64 addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: ds_read_diff_base_interleaving
; CI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-NOT: ds_read_b32
define amdgpu_kernel void @ds_read_diff_base_interleaving(
  float addrspace(1)* nocapture %arg,
  [4 x [4 x float]] addrspace(3)* %arg1,
  [4 x [4 x float]] addrspace(3)* %arg2,
  [4 x [4 x float]] addrspace(3)* %arg3,
  [4 x [4 x float]] addrspace(3)* %arg4) #1 {
bb:
  %tmp = getelementptr float, float addrspace(1)* %arg, i64 10
  %tmp5 = tail call i32 @llvm.amdgcn.workitem.id.x() #2
  %tmp6 = tail call i32 @llvm.amdgcn.workitem.id.y() #2
  %tmp7 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %arg1, i32 0, i32 %tmp6, i32 0
  %tmp8 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %arg2, i32 0, i32 0, i32 %tmp5
  %tmp9 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %arg3, i32 0, i32 %tmp6, i32 0
  %tmp10 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %arg4, i32 0, i32 0, i32 %tmp5
  %tmp11 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %arg1, i32 0, i32 %tmp6, i32 1
  %tmp12 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %arg2, i32 0, i32 1, i32 %tmp5
  %tmp13 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %arg3, i32 0, i32 %tmp6, i32 1
  %tmp14 = getelementptr [4 x [4 x float]], [4 x [4 x float]] addrspace(3)* %arg4, i32 0, i32 1, i32 %tmp5
  %tmp15 = load float, float addrspace(3)* %tmp7
  %tmp16 = load float, float addrspace(3)* %tmp8
  %tmp17 = fmul float %tmp15, %tmp16
  %tmp18 = fadd float 2.000000e+00, %tmp17
  %tmp19 = load float, float addrspace(3)* %tmp9
  %tmp20 = load float, float addrspace(3)* %tmp10
  %tmp21 = fmul float %tmp19, %tmp20
  %tmp22 = fsub float %tmp18, %tmp21
  %tmp23 = load float, float addrspace(3)* %tmp11
  %tmp24 = load float, float addrspace(3)* %tmp12
  %tmp25 = fmul float %tmp23, %tmp24
  %tmp26 = fsub float %tmp22, %tmp25
  %tmp27 = load float, float addrspace(3)* %tmp13
  %tmp28 = load float, float addrspace(3)* %tmp14
  %tmp29 = fmul float %tmp27, %tmp28
  %tmp30 = fsub float %tmp26, %tmp29
  store float %tmp30, float addrspace(1)* %tmp
  ret void
}

; GCN-LABEL: ds_read_call_read:
; GCN: ds_read_b32
; GCN: s_swappc_b64
; GCN: ds_read_b32
define amdgpu_kernel void @ds_read_call_read(i32 addrspace(1)* %out, i32 addrspace(3)* %arg) {
  %x = call i32 @llvm.amdgcn.workitem.id.x()
  %arrayidx0 = getelementptr i32, i32 addrspace(3)* %arg, i32 %x
  %arrayidx1 = getelementptr i32, i32 addrspace(3)* %arrayidx0, i32 1
  %v0 = load i32, i32 addrspace(3)* %arrayidx0, align 4
  call void @void_func_void()
  %v1 = load i32, i32 addrspace(3)* %arrayidx1, align 4
  %r = add i32 %v0, %v1
  store i32 %r, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}ds_read_interp_read:
; CI: s_mov_b32 m0, -1
; CI: ds_read_b32
; CI: s_mov_b32 m0, s0
; CI: v_interp_mov_f32
; CI: s_mov_b32 m0, -1
; CI: ds_read_b32
; GFX9: ds_read2_b32 v[0:1], v0 offset1:4
; GFX9: s_mov_b32 m0, s0
; GFX9: v_interp_mov_f32
define amdgpu_ps <2 x float> @ds_read_interp_read(i32 inreg %prims, float addrspace(3)* %inptr) {
  %v0 = load float, float addrspace(3)* %inptr, align 4
  %intrp = call float @llvm.amdgcn.interp.mov(i32 0, i32 0, i32 0, i32 %prims)
  %ptr1 = getelementptr float, float addrspace(3)* %inptr, i32 4
  %v1 = load float, float addrspace(3)* %ptr1, align 4
  %v1b = fadd float %v1, %intrp
  %r0 = insertelement <2 x float> undef, float %v0, i32 0
  %r1 = insertelement <2 x float> %r0, float %v1b, i32 1
  ret <2 x float> %r1
}

@v2i32_align1 = internal addrspace(3) global [100 x <2 x i32>] undef, align 1

; GCN-LABEL: {{^}}read2_v2i32_align1_odd_offset:
; CI-COUNT-8: ds_read_u8

; GFX9-ALIGNED-COUNT-8: ds_read_u8

; GFX9-UNALIGNED: v_mov_b32_e32 [[BASE_ADDR:v[0-9]+]], 0x41{{$}}
; GFX9-UNALIGNED: ds_read2_b32 v{{\[[0-9]+:[0-9]+\]}}, [[BASE_ADDR]] offset1:1{{$}}
define amdgpu_kernel void @read2_v2i32_align1_odd_offset(<2 x i32> addrspace(1)* %out) {
entry:
  %load = load <2 x i32>, <2 x i32> addrspace(3)* bitcast (i8 addrspace(3)* getelementptr (i8, i8 addrspace(3)* bitcast ([100 x <2 x i32>] addrspace(3)* @v2i32_align1 to i8 addrspace(3)*), i32 65) to <2 x i32> addrspace(3)*), align 1
  store <2 x i32> %load, <2 x i32> addrspace(1)* %out
  ret void
}

declare void @void_func_void() #3

declare i32 @llvm.amdgcn.workgroup.id.x() #1
declare i32 @llvm.amdgcn.workgroup.id.y() #1
declare i32 @llvm.amdgcn.workitem.id.x() #1
declare i32 @llvm.amdgcn.workitem.id.y() #1

declare float @llvm.amdgcn.interp.mov(i32, i32, i32, i32) nounwind readnone

declare void @llvm.amdgcn.s.barrier() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { convergent nounwind }
attributes #3 = { nounwind noinline }
