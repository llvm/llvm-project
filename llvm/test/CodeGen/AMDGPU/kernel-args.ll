; RUN: llc < %s -march=amdgcn -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap -enable-var-scope --check-prefixes=SI,GCN,MESA-GCN,FUNC %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefixes=VI,GCN,MESA-VI,MESA-GCN,FUNC %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx900 --amdhsa-code-object-version=2 -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefixes=VI,GCN,HSA-GFX9,FUNC %s
; RUN: llc < %s -march=r600 -mcpu=redwood -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefixes=EG,EGCM,FUNC %s
; RUN: llc < %s -march=r600 -mcpu=cayman -verify-machineinstrs | FileCheck -allow-deprecated-dag-overlap -enable-var-scope --check-prefixes=CM,EGCM,FUNC %s

; FUNC-LABEL: {{^}}i8_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; SI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; MESA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; MESA-GCN: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff

; HSA-GFX9: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-GFX9: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff


; EGCM: VTX_READ_8{{.*}} #3
; EGCM: KC0[2].Y
define amdgpu_kernel void @i8_arg(i32 addrspace(1)* nocapture %out, i8 %in) nounwind {
  %ext = zext i8 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i8_zext_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c

; HSA-GFX9: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-GFX9: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff


; EG: BFE_INT   T0.X, T0.X, 0.0, literal.x,
; EG-NEXT: LSHR * T1.X, KC0[2].Y, literal.y,
; EG-NEXT: 8(1.121039e-44), 2(2.802597e-45)

; CM: BFE_INT * T0.X, T0.X, 0.0, literal.x,
; CM-NEXT: 8(1.121039e-44), 0(0.000000e+00)
; CM-NEXT: LSHR * T1.X, KC0[2].Y, literal.x,
; CM-NEXT:	2(2.802597e-45), 0(0.000000e+00)
define amdgpu_kernel void @i8_zext_arg(i32 addrspace(1)* nocapture %out, i8 zeroext %in) nounwind {
  %ext = zext i8 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i8_sext_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb

; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c

; HSA-GFX9: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-GFX9: s_sext_i32_i8 s{{[0-9]+}}, [[VAL]]
; HSA-GFX9: global_store_dword


; EG: BFE_INT   T0.X, T0.X, 0.0, literal.x,
; EG-NEXT: LSHR * T1.X, KC0[2].Y, literal.y,
; EG-NEXT: 8(1.121039e-44), 2(2.802597e-45)

; CM: BFE_INT * T0.X, T0.X, 0.0, literal.x,
; CM-NEXT: 8(1.121039e-44), 0(0.000000e+00)
; CM-NEXT: LSHR * T1.X, KC0[2].Y, literal.x,
; CM-NEXT: 2(2.802597e-45), 0(0.000000e+00)
define amdgpu_kernel void @i8_sext_arg(i32 addrspace(1)* nocapture %out, i8 signext %in) nounwind {
  %ext = sext i8 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; SI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb

; MESA-VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; MESA-GCN: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff

; HSA-GFX9: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-GFX9: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xffff{{$}}
; HSA-GFX9: global_store_dword

; EGCM: VTX_READ_16
; EGCM: KC0[2].Y
define amdgpu_kernel void @i16_arg(i32 addrspace(1)* nocapture %out, i16 %in) nounwind {
  %ext = zext i16 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_zext_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c

; HSA-GFX9: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-GFX9: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xffff{{$}}
; HSA-GFX9: global_store_dword

; EG: BFE_INT   T0.X, T0.X, 0.0, literal.x,
; EG-NEXT: LSHR * T1.X, KC0[2].Y, literal.y,
; EG-NEXT: 16(2.242078e-44), 2(2.802597e-45)

; CM: BFE_INT * T0.X, T0.X, 0.0, literal.x,
; CM-NEXT: 16(2.242078e-44), 0(0.000000e+00)
; CM-NEXT: LSHR * T1.X, KC0[2].Y, literal.x,
; CM-NEXT: 2(2.802597e-45), 0(0.000000e+00)
define amdgpu_kernel void @i16_zext_arg(i32 addrspace(1)* nocapture %out, i16 zeroext %in) nounwind {
  %ext = zext i16 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_sext_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c


; HSA-GFX9: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x8
; HSA-GFX9: s_sext_i32_i16 s{{[0-9]+}}, [[VAL]]
; HSA-GFX9: global_store_dword

; EG: BFE_INT   T0.X, T0.X, 0.0, literal.x,
; EG-NEXT: LSHR * T1.X, KC0[2].Y, literal.y,
; EG-NEXT: 16(2.242078e-44), 2(2.802597e-45)

; CM: BFE_INT * T0.X, T0.X, 0.0, literal.x,
; CM-NEXT: 16(2.242078e-44), 0(0.000000e+00)
; CM-NEXT: LSHR * T1.X, KC0[2].Y, literal.x,
; CM-NEXT: 2(2.802597e-45), 0(0.000000e+00)
define amdgpu_kernel void @i16_sext_arg(i32 addrspace(1)* nocapture %out, i16 signext %in) nounwind {
  %ext = sext i16 %in to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; EGCM: T{{[0-9]\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c
; HSA-GFX9: s_load_dword s{{[0-9]}}, s[4:5], 0x8
define amdgpu_kernel void @i32_arg(i32 addrspace(1)* nocapture %out, i32 %in) nounwind {
entry:
  store i32 %in, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}f32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4
; EGCM: T{{[0-9]\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c
; HSA-GFX9: s_load_dword s{{[0-9]+}}, s[4:5], 0x8
define amdgpu_kernel void @f32_arg(float addrspace(1)* nocapture %out, float %in) nounwind {
entry:
  store float %in, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v2i8_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; EGCM: VTX_READ_8
; EGCM: VTX_READ_8

; GCN: s_load_dword s
; GCN-NOT: {{buffer|flat|global}}_load_
define amdgpu_kernel void @v2i8_arg(<2 x i8> addrspace(1)* %out, <2 x i8> %in) {
entry:
  store <2 x i8> %in, <2 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v2i16_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; EGCM: VTX_READ_16
; EGCM: VTX_READ_16

; SI: s_load_dword s{{[0-9]+}}, s[0:1], 0xb
; MESA-VI: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; HSA-GFX9: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x8
define amdgpu_kernel void @v2i16_arg(<2 x i16> addrspace(1)* %out, <2 x i16> %in) {
entry:
  store <2 x i16> %in, <2 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v2i32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 16
; HSA-GFX9: kernarg_segment_alignment = 4

; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[2].W
; SI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xb
; MESA-VI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x2c
; HSA-GFX9: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x8
define amdgpu_kernel void @v2i32_arg(<2 x i32> addrspace(1)* nocapture %out, <2 x i32> %in) nounwind {
entry:
  store <2 x i32> %in, <2 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v2f32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 16
; HSA-GFX9: kernarg_segment_alignment = 4

; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[2].W
; SI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xb
; MESA-VI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x2c
; HSA-GFX9: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[4:5], 0x8
define amdgpu_kernel void @v2f32_arg(<2 x float> addrspace(1)* nocapture %out, <2 x float> %in) nounwind {
entry:
  store <2 x float> %in, <2 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3i8_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; EGCM-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 40
; EGCM-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 41
; EGCM-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 42

; SI: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0xb

; VI-MESA: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; VI-HSA: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x8
define amdgpu_kernel void @v3i8_arg(<3 x i8> addrspace(1)* nocapture %out, <3 x i8> %in) nounwind {
entry:
  store <3 x i8> %in, <3 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3i16_arg:
; HSA-GFX9: kernarg_segment_byte_size = 16
; HSA-GFX9: kernarg_segment_alignment = 4

; EGCM-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 44
; EGCM-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 46
; EGCM-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 48

; SI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xb

; VI-HSA: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x8
; VI-MESA: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x2c
define amdgpu_kernel void @v3i16_arg(<3 x i16> addrspace(1)* nocapture %out, <3 x i16> %in) nounwind {
entry:
  store <3 x i16> %in, <3 x i16> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3i32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 32
; HSA-GFX9: kernarg_segment_alignment = 4
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0xd
; MESA-VI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x34
; HSA-GFX9: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x10
define amdgpu_kernel void @v3i32_arg(<3 x i32> addrspace(1)* nocapture %out, <3 x i32> %in) nounwind {
entry:
  store <3 x i32> %in, <3 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3f32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 32
; HSA-GFX9: kernarg_segment_alignment = 4
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0xd
; MESA-VI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x34
; HSA-GFX9: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x10
define amdgpu_kernel void @v3f32_arg(<3 x float> addrspace(1)* nocapture %out, <3 x float> %in) nounwind {
entry:
  store <3 x float> %in, <3 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v4i8_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8

; GCN-DAG: s_load_dwordx2 s
; GCN-DAG: s_load_dword s
define amdgpu_kernel void @v4i8_arg(<4 x i8> addrspace(1)* %out, <4 x i8> %in) {
entry:
  store <4 x i8> %in, <4 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v4i16_arg:
; HSA-GFX9: kernarg_segment_byte_size = 16
; HSA-GFX9: kernarg_segment_alignment = 4
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16

; SI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0xb
; SI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x9

; MESA-VI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x24
; MESA-VI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x2c


; MESA-VI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x24
; MESA-VI-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x2c

; HSA-GFX9-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x0
; HSA-GFX9-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x8
define amdgpu_kernel void @v4i16_arg(<4 x i16> addrspace(1)* %out, <4 x i16> %in) {
entry:
  store <4 x i16> %in, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v4i32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 32
; HSA-GFX9: kernarg_segment_alignment = 4
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].X

; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xd
; MESA-VI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x34
; HSA-GFX9: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x10
define amdgpu_kernel void @v4i32_arg(<4 x i32> addrspace(1)* nocapture %out, <4 x i32> %in) nounwind {
entry:
  store <4 x i32> %in, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v4f32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 32
; HSA-GFX9: kernarg_segment_alignment = 4
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].X
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xd
; MESA-VI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x34
; HSA-GFX9: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x10
define amdgpu_kernel void @v4f32_arg(<4 x float> addrspace(1)* nocapture %out, <4 x float> %in) nounwind {
entry:
  store <4 x float> %in, <4 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v5i8_arg:
; HSA-GFX9: kernarg_segment_byte_size = 16
; HSA-GFX9: kernarg_segment_alignment = 4

; EGCM-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 46
; EGCM-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 46
; EGCM-DAG: VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 46

; SI: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0xb

; VI-MESA: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; VI-HSA: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x8
define amdgpu_kernel void @v5i8_arg(<5 x i8> addrspace(1)* nocapture %out, <5 x i8> %in) nounwind {
entry:
  store <5 x i8> %in, <5 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v5i16_arg:
; HSA-GFX9: kernarg_segment_byte_size = 32
; HSA-GFX9: kernarg_segment_alignment = 4

; EGCM-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 58
; EGCM-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 58
; EGCM-DAG: VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 58

; SI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xd

; VI-HSA: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x8
; VI-MESA: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x2c
define amdgpu_kernel void @v5i16_arg(<5 x i16> addrspace(1)* nocapture %out, <5 x i16> %in) nounwind {
entry:
  store <5 x i16> %in, <5 x i16> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v5i32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 64
; HSA-GFX9: kernarg_segment_alignment = 5
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].W
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x11
; MESA-VI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x44
; HSA-GFX9: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x20
define amdgpu_kernel void @v5i32_arg(<5 x i32> addrspace(1)* nocapture %out, <5 x i32> %in) nounwind {
entry:
  store <5 x i32> %in, <5 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v5f32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 64
; HSA-GFX9: kernarg_segment_alignment = 5
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].W
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x11
; MESA-VI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x44
; HSA-GFX9: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x20
define amdgpu_kernel void @v5f32_arg(<5 x float> addrspace(1)* nocapture %out, <5 x float> %in) nounwind {
entry:
  store <5 x float> %in, <5 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v5i64_arg:
; HSA-GFX9: kernarg_segment_byte_size = 128
; HSA-GFX9: kernarg_segment_alignment = 6
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Z
; SI-DAG: s_load_dwordx8 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x19
; SI-DAG: s_load_dwordx2 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x21
; MESA-VI-DAG: s_load_dwordx8 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x64
; MESA-VI-DAG: s_load_dwordx2 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x84
; HSA-GFX9-DAG: s_load_dwordx8 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x40
; HSA-GFX9-DAG: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x60
define amdgpu_kernel void @v5i64_arg(<5 x i64> addrspace(1)* nocapture %out, <5 x i64> %in) nounwind {
entry:
  store <5 x i64> %in, <5 x i64> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v5f64_arg:
; HSA-GFX9: kernarg_segment_byte_size = 128
; HSA-GFX9: kernarg_segment_alignment = 6
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Z
; SI-DAG: s_load_dwordx8 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x19
; SI-DAG: s_load_dwordx2 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x21
; MESA-VI-DAG: s_load_dwordx8 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x64
; MESA-VI-DAG: s_load_dwordx2 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x84
; HSA-GFX9-DAG: s_load_dwordx8 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x40
; HSA-GFX9-DAG: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x60
define amdgpu_kernel void @v5f64_arg(<5 x double> addrspace(1)* nocapture %out, <5 x double> %in) nounwind {
entry:
  store <5 x double> %in, <5 x double> addrspace(1)* %out, align 8
  ret void
}

; FIXME: Lots of unpack and re-pack junk on VI
; FUNC-LABEL: {{^}}v8i8_arg:
; HSA-GFX9: kernarg_segment_byte_size = 16
; HSA-GFX9: kernarg_segment_alignment = 4
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8

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
; HSA-GFX9: kernarg_segment_byte_size = 32
; HSA-GFX9: kernarg_segment_alignment = 4
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16

; SI: s_load_dwordx4
; SI-NEXT: s_load_dwordx2
; SI-NOT: {{buffer|flat|global}}_load


; MESA-VI: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x34

; HSA-GFX9: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x10
define amdgpu_kernel void @v8i16_arg(<8 x i16> addrspace(1)* %out, <8 x i16> %in) {
entry:
  store <8 x i16> %in, <8 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v8i32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 64
; HSA-GFX9: kernarg_segment_alignment = 5
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].X

; SI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x11
; MESA-VI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x44
; HSA-GFX9: s_load_dwordx8 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x20
define amdgpu_kernel void @v8i32_arg(<8 x i32> addrspace(1)* nocapture %out, <8 x i32> %in) nounwind {
entry:
  store <8 x i32> %in, <8 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v8f32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 64
; HSA-GFX9: kernarg_segment_alignment = 5
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].X
; SI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x11
define amdgpu_kernel void @v8f32_arg(<8 x float> addrspace(1)* nocapture %out, <8 x float> %in) nounwind {
entry:
  store <8 x float> %in, <8 x float> addrspace(1)* %out, align 4
  ret void
}

; FIXME: Pack/repack on VI

; FUNC-LABEL: {{^}}v16i8_arg:
; HSA-GFX9: kernarg_segment_byte_size = 32
; HSA-GFX9: kernarg_segment_alignment = 4
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8
; EGCM: VTX_READ_8

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
; HSA-GFX9: kernarg_segment_byte_size = 64
; HSA-GFX9: kernarg_segment_alignment = 5
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16

; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16
; EGCM: VTX_READ_16

; SI: s_load_dwordx8 s
; SI-NEXT: s_load_dwordx2 s
; SI-NOT: {{buffer|flat|global}}_load


; MESA-VI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x44

; HSA-GFX9: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x20
define amdgpu_kernel void @v16i16_arg(<16 x i16> addrspace(1)* %out, <16 x i16> %in) {
entry:
  store <16 x i16> %in, <16 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v16i32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 128
; HSA-GFX9: kernarg_segment_alignment = 6
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[10].X
; SI: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x19
; MESA-VI: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x64
; HSA-GFX9: s_load_dwordx16 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x40
define amdgpu_kernel void @v16i32_arg(<16 x i32> addrspace(1)* nocapture %out, <16 x i32> %in) nounwind {
entry:
  store <16 x i32> %in, <16 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v16f32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 128
; HSA-GFX9: kernarg_segment_alignment = 6
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[7].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[8].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].X
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].Y
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].Z
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[9].W
; EGCM-DAG: T{{[0-9]\.[XYZW]}}, KC0[10].X
; SI: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x19
; MESA-VI: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x64
; HSA-GFX9: s_load_dwordx16 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x40
define amdgpu_kernel void @v16f32_arg(<16 x float> addrspace(1)* nocapture %out, <16 x float> %in) nounwind {
entry:
  store <16 x float> %in, <16 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}kernel_arg_i64:
; MESA-VI: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[0:1], 0x24
; HSA-GFX9: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x0

; MESA-GCN: buffer_store_dwordx2
define amdgpu_kernel void @kernel_arg_i64(i64 addrspace(1)* %out, i64 %a) nounwind {
  store i64 %a, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}f64_kernel_arg:
; SI-DAG: s_load_dwordx4 s[{{[0-9]:[0-9]}}], s[0:1], 0x9
; MESA-VI-DAG: s_load_dwordx4 s[{{[0-9]:[0-9]}}], s[0:1], 0x24
; MESA-GCN: buffer_store_dwordx2

; HSA-GFX9: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x0
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

; FUNC-LABEL: {{^}}i65_arg:
; HSA-GFX9: kernarg_segment_byte_size = 24
; HSA-GFX9: kernarg_segment_alignment = 4
; HSA-GFX9: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x0
; HSA-GFX9: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x8
define amdgpu_kernel void @i65_arg(i65 addrspace(1)* nocapture %out, i65 %in) nounwind {
entry:
  store i65 %in, i65 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i1_arg:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; GCN: s_load_dword s
; GCN: s_and_b32
; GCN: {{buffer|flat|global}}_store_byte
define amdgpu_kernel void @i1_arg(i1 addrspace(1)* %out, i1 %x) nounwind {
  store i1 %x, i1 addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_zext_i32:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; GCN: s_load_dword
; GCN: {{buffer|flat|global}}_store_dword
define amdgpu_kernel void @i1_arg_zext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_zext_i64:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; GCN: s_load_dword s
; GCN: {{buffer|flat|global}}_store_dwordx2
define amdgpu_kernel void @i1_arg_zext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_sext_i32:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; GCN: s_load_dword
; GCN: {{buffer|flat|global}}_store_dword
define amdgpu_kernel void @i1_arg_sext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i32
  store i32 %ext, i32addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_sext_i64:
; HSA-GFX9: kernarg_segment_byte_size = 12
; HSA-GFX9: kernarg_segment_alignment = 4

; GCN: s_load_dword
; GCN: s_bfe_i64
; GCN: {{buffer|flat|global}}_store_dwordx2
define amdgpu_kernel void @i1_arg_sext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}empty_struct_arg:
; HSA-GFX9: kernarg_segment_byte_size = 0
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
; HSA-GFX9: kernarg_segment_byte_size = 40
; HSA-GFX9: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; HSA-GFX9: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x8
; HSA-GFX9: s_load_dword s{{[0-9]+}}, s[4:5], 0x18
; HSA-GFX9: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x20
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
; HSA-GFX9: kernarg_segment_byte_size = 28
; HSA-GFX9-DAG: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; HSA-GFX9-DAG: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x4
; HSA-GFX9-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; HSA-GFX9: global_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}} offset:17
; HSA-GFX9: global_load_dword v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}} offset:13
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

; GCN-LABEL: {{^}}struct_argument_alignment_after:
; HSA-GFX9: kernarg_segment_byte_size = 64
; HSA-GFX9: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; HSA-GFX9: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x8
; HSA-GFX9: s_load_dword s{{[0-9]+}}, s[4:5], 0x18
; HSA-GFX9: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x20
; HSA-GFX9: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x30
define amdgpu_kernel void @struct_argument_alignment_after({i32, i64} %arg0, i8, {i32, i64} %arg2, i8, <4 x i32> %arg4) {
  %val0 = extractvalue {i32, i64} %arg0, 0
  %val1 = extractvalue {i32, i64} %arg0, 1
  %val2 = extractvalue {i32, i64} %arg2, 0
  %val3 = extractvalue {i32, i64} %arg2, 1
  store volatile i32 %val0, i32 addrspace(1)* null
  store volatile i64 %val1, i64 addrspace(1)* null
  store volatile i32 %val2, i32 addrspace(1)* null
  store volatile i64 %val3, i64 addrspace(1)* null
  store volatile <4 x i32> %arg4, <4 x i32> addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}array_3xi32:
; HSA-GFX9: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; HSA-GFX9: s_load_dword s{{[0-9]+}}, s[4:5], 0x4
; HSA-GFX9: s_load_dword s{{[0-9]+}}, s[4:5], 0x8
; HSA-GFX9: s_load_dword s{{[0-9]+}}, s[4:5], 0xc
define amdgpu_kernel void @array_3xi32(i16 %arg0, [3 x i32] %arg1) {
  store volatile i16 %arg0, i16 addrspace(1)* undef
  store volatile [3 x i32] %arg1, [3 x i32] addrspace(1)* undef
  ret void
}

; FIXME: Why not all scalar loads?
; GCN-LABEL: {{^}}array_3xi16:
; HSA-GFX9-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; HSA-GFX9: global_load_ushort v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}} offset:2
; HSA-GFX9: global_load_ushort v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}} offset:4
; HSA-GFX9: global_load_ushort v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}} offset:6
define amdgpu_kernel void @array_3xi16(i8 %arg0, [3 x i16] %arg1) {
  store volatile i8 %arg0, i8 addrspace(1)* undef
  store volatile [3 x i16] %arg1, [3 x i16] addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}small_array_round_down_offset:
; HSA-GFX9-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; HSA-GFX9: global_load_ubyte v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}} offset:1
define amdgpu_kernel void @small_array_round_down_offset(i8, [1 x i8] %arg) {
  %val = extractvalue [1 x i8] %arg, 0
  store volatile i8 %val, i8 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}byref_align_constant_i32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 264
; HSA-GFX9-DAG: s_load_dwordx2 {{s\[[0-9]+:[0-9]+\]}}, s[4:5], 0x100{{$}}
define amdgpu_kernel void @byref_align_constant_i32_arg(i32 addrspace(1)* nocapture %out, i32 addrspace(4)* byref(i32) align(256) %in.byref, i32 %after.offset) {
  %in = load i32, i32 addrspace(4)* %in.byref
  store volatile i32 %in, i32 addrspace(1)* %out, align 4
  store volatile i32 %after.offset, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}byref_natural_align_constant_v16i32_arg:
; HSA-GFX9: kernarg_segment_byte_size = 132
; HSA-GFX9-DAG: s_load_dword s{{[0-9]+}}, s[4:5], 0x80
; HSA-GFX9-DAG: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x40{{$}}
define amdgpu_kernel void @byref_natural_align_constant_v16i32_arg(i32 addrspace(1)* nocapture %out, i8, <16 x i32> addrspace(4)* byref(<16 x i32>) %in.byref, i32 %after.offset) {
  %in = load <16 x i32>, <16 x i32> addrspace(4)* %in.byref
  %cast.out = bitcast i32 addrspace(1)* %out to <16 x i32> addrspace(1)*
  store volatile <16 x i32> %in, <16 x i32> addrspace(1)* %cast.out, align 4
  store volatile i32 %after.offset, i32 addrspace(1)* %out, align 4
  ret void
}
