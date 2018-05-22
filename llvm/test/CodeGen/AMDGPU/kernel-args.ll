; RUN: llc < %s -march=amdgcn -verify-machineinstrs | FileCheck -enable-var-scope --check-prefixes=SI,GCN,MESA-GCN,FUNC %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck -enable-var-scope -check-prefixes=VI,GCN,MESA-VI,MESA-GCN,FUNC %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs | FileCheck -enable-var-scope -check-prefixes=VI,GCN,HSA-VI,FUNC %s
; RUN: llc < %s -march=r600 -mcpu=redwood -verify-machineinstrs | FileCheck -enable-var-scope -check-prefix=EG --check-prefix=FUNC %s
; RUN: llc < %s -march=r600 -mcpu=cayman -verify-machineinstrs | FileCheck -enable-var-scope --check-prefix=EG --check-prefix=FUNC %s

; FUNC-LABEL: {{^}}i8_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: AND_INT {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; MESA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; MESA-GCN: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff
; HSA-VI: s_add_u32 [[SPTR_LO:s[0-9]+]], s4, 8
; HSA-VI: s_addc_u32 [[SPTR_HI:s[0-9]+]], s5, 0
; HSA-VI: v_mov_b32_e32 v[[VPTR_LO:[0-9]+]], [[SPTR_LO]]
; HSA-VI: v_mov_b32_e32 v[[VPTR_HI:[0-9]+]], [[SPTR_HI]]
; FIXME: Should be using s_load_dword
; HSA-VI: flat_load_ubyte v{{[0-9]+}}, v{{\[}}[[VPTR_LO]]:[[VPTR_HI]]]{{$}}

define amdgpu_kernel void @i8_arg(i32 addrspace(1)* nocapture %out, i8 %in) nounwind {
entry:
  %0 = zext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i8_zext_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c
; HSA-VI: s_add_u32 [[SPTR_LO:s[0-9]+]], s4, 8
; HSA-VI: s_addc_u32 [[SPTR_HI:s[0-9]+]], s5, 0
; HSA-VI: v_mov_b32_e32 v[[VPTR_LO:[0-9]+]], [[SPTR_LO]]
; HSA-VI: v_mov_b32_e32 v[[VPTR_HI:[0-9]+]], [[SPTR_HI]]
; FIXME: Should be using s_load_dword
; HSA-VI: flat_load_ubyte v{{[0-9]+}}, v{{\[}}[[VPTR_LO]]:[[VPTR_HI]]]{{$}}

define amdgpu_kernel void @i8_zext_arg(i32 addrspace(1)* nocapture %out, i8 zeroext %in) nounwind {
entry:
  %0 = zext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i8_sext_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c
; HSA-VI: s_add_u32 [[SPTR_LO:s[0-9]+]], s4, 8
; HSA-VI: s_addc_u32 [[SPTR_HI:s[0-9]+]], s5, 0
; HSA-VI: v_mov_b32_e32 v[[VPTR_LO:[0-9]+]], [[SPTR_LO]]
; HSA-VI: v_mov_b32_e32 v[[VPTR_HI:[0-9]+]], [[SPTR_HI]]
; FIXME: Should be using s_load_dword
; HSA-VI: flat_load_sbyte v{{[0-9]+}}, v{{\[}}[[VPTR_LO]]:[[VPTR_HI]]]{{$}}

define amdgpu_kernel void @i8_sext_arg(i32 addrspace(1)* nocapture %out, i8 signext %in) nounwind {
entry:
  %0 = sext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: AND_INT {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; MESA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; MESA-GCN: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff
; HSA-VI: s_add_u32 [[SPTR_LO:s[0-9]+]], s4, 8
; HSA-VI: s_addc_u32 [[SPTR_HI:s[0-9]+]], s5, 0
; HSA-VI: v_mov_b32_e32 v[[VPTR_LO:[0-9]+]], [[SPTR_LO]]
; HSA-VI: v_mov_b32_e32 v[[VPTR_HI:[0-9]+]], [[SPTR_HI]]
; FIXME: Should be using s_load_dword
; HSA-VI: flat_load_ushort v{{[0-9]+}}, v{{\[}}[[VPTR_LO]]:[[VPTR_HI]]]{{$}}

define amdgpu_kernel void @i16_arg(i32 addrspace(1)* nocapture %out, i16 %in) nounwind {
entry:
  %0 = zext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_zext_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c
; HSA-VI: s_add_u32 [[SPTR_LO:s[0-9]+]], s4, 8
; HSA-VI: s_addc_u32 [[SPTR_HI:s[0-9]+]], s5, 0
; HSA-VI: v_mov_b32_e32 v[[VPTR_LO:[0-9]+]], [[SPTR_LO]]
; HSA-VI: v_mov_b32_e32 v[[VPTR_HI:[0-9]+]], [[SPTR_HI]]
; FIXME: Should be using s_load_dword
; HSA-VI: flat_load_ushort v{{[0-9]+}}, v{{\[}}[[VPTR_LO]]:[[VPTR_HI]]]{{$}}

define amdgpu_kernel void @i16_zext_arg(i32 addrspace(1)* nocapture %out, i16 zeroext %in) nounwind {
entry:
  %0 = zext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_sext_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c
; HSA-VI: s_add_u32 [[SPTR_LO:s[0-9]+]], s4, 8
; HSA-VI: s_addc_u32 [[SPTR_HI:s[0-9]+]], s5, 0
; HSA-VI: v_mov_b32_e32 v[[VPTR_LO:[0-9]+]], [[SPTR_LO]]
; HSA-VI: v_mov_b32_e32 v[[VPTR_HI:[0-9]+]], [[SPTR_HI]]
; FIXME: Should be using s_load_dword
; HSA-VI: flat_load_sshort v{{[0-9]+}}, v{{\[}}[[VPTR_LO]]:[[VPTR_HI]]]{{$}}

define amdgpu_kernel void @i16_sext_arg(i32 addrspace(1)* nocapture %out, i16 signext %in) nounwind {
entry:
  %0 = sext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i32_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: T{{[0-9]\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c
; HSA-VI: s_load_dword s{{[0-9]}}, s[4:5], 0x8
define amdgpu_kernel void @i32_arg(i32 addrspace(1)* nocapture %out, i32 %in) nounwind {
entry:
  store i32 %in, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}f32_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: T{{[0-9]\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x8
define amdgpu_kernel void @f32_arg(float addrspace(1)* nocapture %out, float %in) nounwind {
entry:
  store float %in, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v2i8_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: VTX_READ_8
; EG: VTX_READ_8
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
define amdgpu_kernel void @v2i8_arg(<2 x i8> addrspace(1)* %out, <2 x i8> %in) {
entry:
  store <2 x i8> %in, <2 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v2i16_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: VTX_READ_16
; EG: VTX_READ_16

; SI: buffer_load_ushort
; SI: buffer_load_ushort

; VI: s_load_dword s
define amdgpu_kernel void @v2i16_arg(<2 x i16> addrspace(1)* %out, <2 x i16> %in) {
entry:
  store <2 x i16> %in, <2 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v2i32_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[2].W
; SI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xb
; MESA-VI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x2c
; HSA-VI: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x8
define amdgpu_kernel void @v2i32_arg(<2 x i32> addrspace(1)* nocapture %out, <2 x i32> %in) nounwind {
entry:
  store <2 x i32> %in, <2 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v2f32_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[2].W
; SI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xb
; MESA-VI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x2c
; HSA-VI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[4:5], 0x8
define amdgpu_kernel void @v2f32_arg(<2 x float> addrspace(1)* nocapture %out, <2 x float> %in) nounwind {
entry:
  store <2 x float> %in, <2 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3i8_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 40
; EG-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 41
; EG-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 42
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
define amdgpu_kernel void @v3i8_arg(<3 x i8> addrspace(1)* nocapture %out, <3 x i8> %in) nounwind {
entry:
  store <3 x i8> %in, <3 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3i16_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 44
; EG-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 46
; EG-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 48
; MESA-GCN: buffer_load_ushort
; MESA-GCN: buffer_load_ushort
; MESA-GCN: buffer_load_ushort
; HSA-VI: flat_load_ushort
; HSA-VI: flat_load_ushort
; HSA-VI: flat_load_ushort
define amdgpu_kernel void @v3i16_arg(<3 x i16> addrspace(1)* nocapture %out, <3 x i16> %in) nounwind {
entry:
  store <3 x i16> %in, <3 x i16> addrspace(1)* %out, align 4
  ret void
}
; FUNC-LABEL: {{^}}v3i32_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0xd
; MESA-VI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x34
; HSA-VI: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x10
define amdgpu_kernel void @v3i32_arg(<3 x i32> addrspace(1)* nocapture %out, <3 x i32> %in) nounwind {
entry:
  store <3 x i32> %in, <3 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3f32_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0xd
; MESA-VI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x34
; HSA-VI: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x10
define amdgpu_kernel void @v3f32_arg(<3 x float> addrspace(1)* nocapture %out, <3 x float> %in) nounwind {
entry:
  store <3 x float> %in, <3 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v4i8_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
define amdgpu_kernel void @v4i8_arg(<4 x i8> addrspace(1)* %out, <4 x i8> %in) {
entry:
  store <4 x i8> %in, <4 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v4i16_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16

; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort

; VI: s_load_dword s
; VI: s_load_dword s
define amdgpu_kernel void @v4i16_arg(<4 x i16> addrspace(1)* %out, <4 x i16> %in) {
entry:
  store <4 x i16> %in, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v4i32_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].X

; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xd
; MESA-VI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x34
; HSA-VI: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x10
define amdgpu_kernel void @v4i32_arg(<4 x i32> addrspace(1)* nocapture %out, <4 x i32> %in) nounwind {
entry:
  store <4 x i32> %in, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v4f32_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].X
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xd
; MESA-VI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x34
; HSA-VI: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x10
define amdgpu_kernel void @v4f32_arg(<4 x float> addrspace(1)* nocapture %out, <4 x float> %in) nounwind {
entry:
  store <4 x float> %in, <4 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v8i8_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; HSA-GCN: float_load_ubyte
; HSA-GCN: float_load_ubyte
; HSA-GCN: float_load_ubyte
; HSA-GCN: float_load_ubyte
; HSA-GCN: float_load_ubyte
; HSA-GCN: float_load_ubyte
; HSA-GCN: float_load_ubyte
; HSA-GCN: float_load_ubyte
define amdgpu_kernel void @v8i8_arg(<8 x i8> addrspace(1)* %out, <8 x i8> %in) {
entry:
  store <8 x i8> %in, <8 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v8i16_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16

; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort

; VI: s_load_dword s
; VI: s_load_dword s
; VI: s_load_dword s
; VI: s_load_dword s
define amdgpu_kernel void @v8i16_arg(<8 x i16> addrspace(1)* %out, <8 x i16> %in) {
entry:
  store <8 x i16> %in, <8 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v8i32_arg:
; HSA-VI: kernarg_segment_alignment = 5
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].X
; SI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x11
; MESA-VI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x44
; HSA-VI: s_load_dwordx8 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x20
define amdgpu_kernel void @v8i32_arg(<8 x i32> addrspace(1)* nocapture %out, <8 x i32> %in) nounwind {
entry:
  store <8 x i32> %in, <8 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v8f32_arg:
; HSA-VI: kernarg_segment_alignment = 5
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].X
; SI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x11
define amdgpu_kernel void @v8f32_arg(<8 x float> addrspace(1)* nocapture %out, <8 x float> %in) nounwind {
entry:
  store <8 x float> %in, <8 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v16i8_arg:
; HSA-VI: kernarg_segment_alignment = 4
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; MESA-GCN: buffer_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
; HSA-VI: flat_load_ubyte
define amdgpu_kernel void @v16i8_arg(<16 x i8> addrspace(1)* %out, <16 x i8> %in) {
entry:
  store <16 x i8> %in, <16 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v16i16_arg:
; HSA-VI: kernarg_segment_alignment = 5
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16

; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_load_ushort

; VI: s_load_dword s
; VI: s_load_dword s
; VI: s_load_dword s
; VI: s_load_dword s
; VI: s_load_dword s
; VI: s_load_dword s
; VI: s_load_dword s
; VI: s_load_dword s
define amdgpu_kernel void @v16i16_arg(<16 x i16> addrspace(1)* %out, <16 x i16> %in) {
entry:
  store <16 x i16> %in, <16 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v16i32_arg:
; HSA-VI: kernarg_segment_alignment = 6
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[10].X
; SI: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x19
; MESA-VI: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x64
; HSA-VI: s_load_dwordx16 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x40
define amdgpu_kernel void @v16i32_arg(<16 x i32> addrspace(1)* nocapture %out, <16 x i32> %in) nounwind {
entry:
  store <16 x i32> %in, <16 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v16f32_arg:
; HSA-VI: kernarg_segment_alignment = 6
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[10].X
; SI: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x19
; MESA-VI: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x64
; HSA-VI: s_load_dwordx16 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x40
define amdgpu_kernel void @v16f32_arg(<16 x float> addrspace(1)* nocapture %out, <16 x float> %in) nounwind {
entry:
  store <16 x float> %in, <16 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}kernel_arg_i64:
; MESA-GCN: s_load_dwordx2
; MESA-GCN: s_load_dwordx2
; MESA-GCN: buffer_store_dwordx2
; HSA-VI: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x8
define amdgpu_kernel void @kernel_arg_i64(i64 addrspace(1)* %out, i64 %a) nounwind {
  store i64 %a, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}f64_kernel_arg:
; SI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0x9
; SI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0xb
; MESA-VI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0x24
; MESA-VI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0x2c
; MESA-GCN: buffer_store_dwordx2
; HSA-VI: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x8
define amdgpu_kernel void @f64_kernel_arg(double addrspace(1)* %out, double  %in) {
entry:
  store double %in, double addrspace(1)* %out
  ret void
}

; XFUNC-LABEL: {{^}}kernel_arg_v1i64:
; XGCN: s_load_dwordx2
; XGCN: s_load_dwordx2
; XGCN: buffer_store_dwordx2
; define amdgpu_kernel void @kernel_arg_v1i64(<1 x i64> addrspace(1)* %out, <1 x i64> %a) nounwind {
;   store <1 x i64> %a, <1 x i64> addrspace(1)* %out, align 8
;   ret void
; }

; FUNC-LABEL: {{^}}i1_arg:
; SI: buffer_load_ubyte
; SI: v_and_b32_e32
; SI: buffer_store_byte
; SI: s_endpgm
define amdgpu_kernel void @i1_arg(i1 addrspace(1)* %out, i1 %x) nounwind {
  store i1 %x, i1 addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_zext_i32:
; SI: buffer_load_ubyte
; SI: buffer_store_dword
; SI: s_endpgm
define amdgpu_kernel void @i1_arg_zext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_zext_i64:
; SI: buffer_load_ubyte
; SI: buffer_store_dwordx2
; SI: s_endpgm
define amdgpu_kernel void @i1_arg_zext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_sext_i32:
; SI: buffer_load_ubyte
; SI: buffer_store_dword
; SI: s_endpgm
define amdgpu_kernel void @i1_arg_sext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i32
  store i32 %ext, i32addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_sext_i64:
; SI: buffer_load_ubyte
; SI: v_bfe_i32
; SI: v_ashrrev_i32
; SI: buffer_store_dwordx2
; SI: s_endpgm
define amdgpu_kernel void @i1_arg_sext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}
