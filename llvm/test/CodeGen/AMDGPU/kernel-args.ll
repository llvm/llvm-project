; RUN: llc < %s -march=amdgcn -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap -enable-var-scope --check-prefixes=SI,GCN,MESA-GCN,FUNC %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefixes=VI,GCN,MESA-VI,MESA-GCN,FUNC %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefixes=VI,GCN,HSA-VI,FUNC %s
; RUN: llc < %s -march=r600 -mcpu=redwood -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefix=EG --check-prefix=FUNC %s
; RUN: llc < %s -march=r600 -mcpu=cayman -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap -enable-var-scope --check-prefix=EG --check-prefix=FUNC %s

; FUNC-LABEL: {{^}}i8_arg:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4
; EG: AND_INT {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; MESA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; MESA-GCN: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff

; HSA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-VI: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff
define amdgpu_kernel void @i8_arg(i32 addrspace(1)* nocapture %out, i8 %in) nounwind {
  %ext = zext i8 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i8_zext_arg:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4
; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c

; HSA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-VI: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff
define amdgpu_kernel void @i8_zext_arg(i32 addrspace(1)* nocapture %out, i8 zeroext %in) nounwind {
  %ext = zext i8 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i8_sext_arg:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4
; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb

; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c

; HSA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-VI: s_sext_i32_i8 s{{[0-9]+}}, [[VAL]]
; HSA-VI: flat_store_dword
define amdgpu_kernel void @i8_sext_arg(i32 addrspace(1)* nocapture %out, i8 signext %in) nounwind {
  %ext = sext i8 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_arg:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; EG: AND_INT {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb

; MESA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; MESA-GCN: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff

; HSA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-VI: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xffff{{$}}
; HSA-VI: flat_store_dword
define amdgpu_kernel void @i16_arg(i32 addrspace(1)* nocapture %out, i16 %in) nounwind {
  %ext = zext i16 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_zext_arg:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c

; HSA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-VI: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xffff{{$}}
; HSA-VI: flat_store_dword
define amdgpu_kernel void @i16_zext_arg(i32 addrspace(1)* nocapture %out, i16 zeroext %in) nounwind {
  %ext = zext i16 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_sext_arg:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c


; HSA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-VI: s_sext_i32_i16 s{{[0-9]+}}, [[VAL]]
; HSA-VI: flat_store_dword
define amdgpu_kernel void @i16_sext_arg(i32 addrspace(1)* nocapture %out, i16 signext %in) nounwind {
  %ext = sext i16 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i32_arg:
; HSA-VI: kernarg_segment_byte_size = 12
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
; HSA-VI: kernarg_segment_byte_size = 12
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
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; EG: VTX_READ_8
; EG: VTX_READ_8

; GCN: s_load_dword s
; GCN-NOT: {{buffer|flat|global}}_load_
define amdgpu_kernel void @v2i8_arg(<2 x i8> addrspace(1)* %out, <2 x i8> %in) {
entry:
  store <2 x i8> %in, <2 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v2i16_arg:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; EG: VTX_READ_16
; EG: VTX_READ_16

; SI: s_load_dword s{{[0-9]+}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; HSA-VI: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x8
define amdgpu_kernel void @v2i16_arg(<2 x i16> addrspace(1)* %out, <2 x i16> %in) {
entry:
  store <2 x i16> %in, <2 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v2i32_arg:
; HSA-VI: kernarg_segment_byte_size = 16
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
; HSA-VI: kernarg_segment_byte_size = 16
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
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; EG-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 40
; EG-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 41
; EG-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 42

; SI: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0xb

; VI-MESA: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; VI-HSA: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x8
define amdgpu_kernel void @v3i8_arg(<3 x i8> addrspace(1)* nocapture %out, <3 x i8> %in) nounwind {
entry:
  store <3 x i8> %in, <3 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3i16_arg:
; HSA-VI: kernarg_segment_byte_size = 16
; HSA-VI: kernarg_segment_alignment = 4

; EG-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 44
; EG-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 46
; EG-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 48

; SI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xb

; VI-HSA: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x8
; VI-MESA: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x2c
define amdgpu_kernel void @v3i16_arg(<3 x i16> addrspace(1)* nocapture %out, <3 x i16> %in) nounwind {
entry:
  store <3 x i16> %in, <3 x i16> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3i32_arg:
; HSA-VI: kernarg_segment_byte_size = 32
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
; HSA-VI: kernarg_segment_byte_size = 32
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
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8

; GCN-DAG: s_load_dwordx2 s
; GCN-DAG: s_load_dword s
define amdgpu_kernel void @v4i8_arg(<4 x i8> addrspace(1)* %out, <4 x i8> %in) {
entry:
  store <4 x i8> %in, <4 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v4i16_arg:
; HSA-VI: kernarg_segment_byte_size = 16
; HSA-VI: kernarg_segment_alignment = 4
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16

; SI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0xb
; SI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x9

; MESA-VI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x24
; MESA-VI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x2c


; MESA-VI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x24
; MESA-VI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x2c

; HSA-VI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x0
; HSA-VI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x8
define amdgpu_kernel void @v4i16_arg(<4 x i16> addrspace(1)* %out, <4 x i16> %in) {
entry:
  store <4 x i16> %in, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v4i32_arg:
; HSA-VI: kernarg_segment_byte_size = 32
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
; HSA-VI: kernarg_segment_byte_size = 32
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

; FIXME: Lots of unpack and re-pack junk on VI
; FUNC-LABEL: {{^}}v8i8_arg:
; HSA-VI: kernarg_segment_byte_size = 16
; HSA-VI: kernarg_segment_alignment = 4
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8

; SI-NOT: {{buffer|flat|global}}_load
; SI: s_load_dwordx2 s
; SI-NEXT: s_load_dwordx2 s
; SI-NOT: {{buffer|flat|global}}_load

; VI: s_load_dwordx2 s
; VI-NEXT: s_load_dwordx2 s
; VI-NOT: lshl
; VI-NOT: _or
; VI-NOT: _sdwa
define amdgpu_kernel void @v8i8_arg(<8 x i8> addrspace(1)* %out, <8 x i8> %in) {
entry:
  store <8 x i8> %in, <8 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v8i16_arg:
; HSA-VI: kernarg_segment_byte_size = 32
; HSA-VI: kernarg_segment_alignment = 4
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16

; SI: s_load_dwordx4
; SI-NEXT: s_load_dwordx2
; SI-NOT: {{buffer|flat|global}}_load


; MESA-VI: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x34

; HSA-VI: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x10
define amdgpu_kernel void @v8i16_arg(<8 x i16> addrspace(1)* %out, <8 x i16> %in) {
entry:
  store <8 x i16> %in, <8 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v8i32_arg:
; HSA-VI: kernarg_segment_byte_size = 64
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
; HSA-VI: kernarg_segment_byte_size = 64
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

; FIXME: Pack/repack on VI

; FUNC-LABEL: {{^}}v16i8_arg:
; HSA-VI: kernarg_segment_byte_size = 32
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

; SI: s_load_dwordx4 s
; SI-NEXT: s_load_dwordx2 s
; SI-NOT: {{buffer|flat|global}}_load


; VI: s_load_dwordx4 s
; VI-NOT: shr
; VI-NOT: shl
; VI-NOT: _sdwa
; VI-NOT: _or_
define amdgpu_kernel void @v16i8_arg(<16 x i8> addrspace(1)* %out, <16 x i8> %in) {
entry:
  store <16 x i8> %in, <16 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v16i16_arg:
; HSA-VI: kernarg_segment_byte_size = 64
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

; SI: s_load_dwordx8 s
; SI-NEXT: s_load_dwordx2 s
; SI-NOT: {{buffer|flat|global}}_load


; MESA-VI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x44

; HSA-VI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x20
define amdgpu_kernel void @v16i16_arg(<16 x i16> addrspace(1)* %out, <16 x i16> %in) {
entry:
  store <16 x i16> %in, <16 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v16i32_arg:
; HSA-VI: kernarg_segment_byte_size = 128
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
; HSA-VI: kernarg_segment_byte_size = 128
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
; MESA-VI: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[0:1], 0x24
; HSA-VI: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x0

; MESA-GCN: buffer_store_dwordx2
define amdgpu_kernel void @kernel_arg_i64(i64 addrspace(1)* %out, i64 %a) nounwind {
  store i64 %a, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}f64_kernel_arg:
; SI-DAG: s_load_dwordx4 s[{{[0-9]:[0-9]}}], s[0:1], 0x9
; MESA-VI-DAG: s_load_dwordx4 s[{{[0-9]:[0-9]}}], s[0:1], 0x24
; MESA-GCN: buffer_store_dwordx2

; HSA-VI: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x0
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
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; GCN: s_load_dword s
; GCN: s_and_b32
; GCN: {{buffer|flat}}_store_byte
define amdgpu_kernel void @i1_arg(i1 addrspace(1)* %out, i1 %x) nounwind {
  store i1 %x, i1 addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_zext_i32:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; GCN: s_load_dword
; SGCN: buffer_store_dword
define amdgpu_kernel void @i1_arg_zext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_zext_i64:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; GCN: s_load_dword s
; GCN: {{buffer|flat}}_store_dwordx2
define amdgpu_kernel void @i1_arg_zext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_sext_i32:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; GCN: s_load_dword
; GCN: {{buffer|flat}}_store_dword
define amdgpu_kernel void @i1_arg_sext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i32
  store i32 %ext, i32addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_sext_i64:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; GCN: s_load_dword
; GCN: s_bfe_i64
; GCN: {{buffer|flat}}_store_dwordx2
define amdgpu_kernel void @i1_arg_sext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}empty_struct_arg:
; HSA: kernarg_segment_byte_size = 0
define amdgpu_kernel void @empty_struct_arg({} %in) nounwind {
  ret void
}

; The correct load offsets for these:
; load 4 from 0,
; load 8 from 8
; load 4 from 24
; load 8 from 32

; With the SelectionDAG argument lowering, the alignments for the
; struct members is not properly considered, making these wrong.

; FIXME: Total argument size is computed wrong
; FUNC-LABEL: {{^}}struct_argument_alignment:
; HSA: kernarg_segment_byte_size = 40
; HSA: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; HSA: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x8
; HSA: s_load_dword s{{[0-9]+}}, s[4:5], 0x18
; HSA: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x20
define amdgpu_kernel void @struct_argument_alignment({i32, i64} %arg0, i8, {i32, i64} %arg1) {
  %val0 = extractvalue {i32, i64} %arg0, 0
  %val1 = extractvalue {i32, i64} %arg0, 1
  %val2 = extractvalue {i32, i64} %arg1, 0
  %val3 = extractvalue {i32, i64} %arg1, 1
  store volatile i32 %val0, i32 addrspace(1)* null
  store volatile i64 %val1, i64 addrspace(1)* null
  store volatile i32 %val2, i32 addrspace(1)* null
  store volatile i64 %val3, i64 addrspace(1)* null
  ret void
}

; No padding between i8 and next struct, but round up at end to 4 byte
; multiple.
; FUNC-LABEL: {{^}}packed_struct_argument_alignment:
; HSA: kernarg_segment_byte_size = 28
; HSA: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; HSA: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x4
; HSA: s_load_dword s{{[0-9]+}}, s[4:5], 0xc
; HSA: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x10
define amdgpu_kernel void @packed_struct_argument_alignment(<{i32, i64}> %arg0, i8, <{i32, i64}> %arg1) {
  %val0 = extractvalue <{i32, i64}> %arg0, 0
  %val1 = extractvalue <{i32, i64}> %arg0, 1
  %val2 = extractvalue <{i32, i64}> %arg1, 0
  %val3 = extractvalue <{i32, i64}> %arg1, 1
  store volatile i32 %val0, i32 addrspace(1)* null
  store volatile i64 %val1, i64 addrspace(1)* null
  store volatile i32 %val2, i32 addrspace(1)* null
  store volatile i64 %val3, i64 addrspace(1)* null
  ret void
}
