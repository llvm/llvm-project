; RUN: llc < %s -march=amdgcn -verify-machineinstrs | FileCheck %s --check-prefix=SI --check-prefix=GCN --check-prefix=FUNC
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s --check-prefix=VI --check-prefix=GCN --check-prefix=FUNC
; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=EG --check-prefix=FUNC
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s --check-prefix=EG --check-prefix=FUNC

; FUNC-LABEL: {{^}}i8_arg:
; EG: AND_INT {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; GCN: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff

define void @i8_arg(i32 addrspace(1)* nocapture %out, i8 %in) nounwind {
entry:
  %0 = zext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i8_zext_arg:
; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c

define void @i8_zext_arg(i32 addrspace(1)* nocapture %out, i8 zeroext %in) nounwind {
entry:
  %0 = zext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i8_sext_arg:
; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c

define void @i8_sext_arg(i32 addrspace(1)* nocapture %out, i8 signext %in) nounwind {
entry:
  %0 = sext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_arg:
; EG: AND_INT {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; GCN: s_and_b32 s{{[0-9]+}}, [[VAL]], 0xff

define void @i16_arg(i32 addrspace(1)* nocapture %out, i16 %in) nounwind {
entry:
  %0 = zext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_zext_arg:
; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c

define void @i16_zext_arg(i32 addrspace(1)* nocapture %out, i16 zeroext %in) nounwind {
entry:
  %0 = zext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i16_sext_arg:
; EG: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c

define void @i16_sext_arg(i32 addrspace(1)* nocapture %out, i16 signext %in) nounwind {
entry:
  %0 = sext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i32_arg:
; EG: T{{[0-9]\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c
define void @i32_arg(i32 addrspace(1)* nocapture %out, i32 %in) nounwind {
entry:
  store i32 %in, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}f32_arg:
; EG: T{{[0-9]\.[XYZW]}}, KC0[2].Z
; SI: s_load_dword s{{[0-9]}}, s[0:1], 0xb
; VI: s_load_dword s{{[0-9]}}, s[0:1], 0x2c
define void @f32_arg(float addrspace(1)* nocapture %out, float %in) nounwind {
entry:
  store float %in, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v2i8_arg:
; EG: VTX_READ_8
; EG: VTX_READ_8
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
define void @v2i8_arg(<2 x i8> addrspace(1)* %out, <2 x i8> %in) {
entry:
  store <2 x i8> %in, <2 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v2i16_arg:
; EG: VTX_READ_16
; EG: VTX_READ_16
; GCN-DAG: buffer_load_ushort
; GCN-DAG: buffer_load_ushort
define void @v2i16_arg(<2 x i16> addrspace(1)* %out, <2 x i16> %in) {
entry:
  store <2 x i16> %in, <2 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v2i32_arg:
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[2].W
; SI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xb
; VI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x2c
define void @v2i32_arg(<2 x i32> addrspace(1)* nocapture %out, <2 x i32> %in) nounwind {
entry:
  store <2 x i32> %in, <2 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v2f32_arg:
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[2].W
; SI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xb
; VI: s_load_dwordx2 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x2c
define void @v2f32_arg(<2 x float> addrspace(1)* nocapture %out, <2 x float> %in) nounwind {
entry:
  store <2 x float> %in, <2 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3i8_arg:
; VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 40
; VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 41
; VTX_READ_8 T{{[0-9]}}.X, T{{[0-9]}}.X, 42
define void @v3i8_arg(<3 x i8> addrspace(1)* nocapture %out, <3 x i8> %in) nounwind {
entry:
  store <3 x i8> %in, <3 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3i16_arg:
; VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 44
; VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 46
; VTX_READ_16 T{{[0-9]}}.X, T{{[0-9]}}.X, 48
define void @v3i16_arg(<3 x i16> addrspace(1)* nocapture %out, <3 x i16> %in) nounwind {
entry:
  store <3 x i16> %in, <3 x i16> addrspace(1)* %out, align 4
  ret void
}
; FUNC-LABEL: {{^}}v3i32_arg:
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0xd
; VI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x34
define void @v3i32_arg(<3 x i32> addrspace(1)* nocapture %out, <3 x i32> %in) nounwind {
entry:
  store <3 x i32> %in, <3 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v3f32_arg:
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0xd
; VI: s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x34
define void @v3f32_arg(<3 x float> addrspace(1)* nocapture %out, <3 x float> %in) nounwind {
entry:
  store <3 x float> %in, <3 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v4i8_arg:
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
define void @v4i8_arg(<4 x i8> addrspace(1)* %out, <4 x i8> %in) {
entry:
  store <4 x i8> %in, <4 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v4i16_arg:
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
define void @v4i16_arg(<4 x i16> addrspace(1)* %out, <4 x i16> %in) {
entry:
  store <4 x i16> %in, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v4i32_arg:
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].X
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xd
; VI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x34
define void @v4i32_arg(<4 x i32> addrspace(1)* nocapture %out, <4 x i32> %in) nounwind {
entry:
  store <4 x i32> %in, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v4f32_arg:
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[3].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].X
; SI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0xd
; VI: s_load_dwordx4 s{{\[[0-9]:[0-9]\]}}, s[0:1], 0x34
define void @v4f32_arg(<4 x float> addrspace(1)* nocapture %out, <4 x float> %in) nounwind {
entry:
  store <4 x float> %in, <4 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v8i8_arg:
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
define void @v8i8_arg(<8 x i8> addrspace(1)* %out, <8 x i8> %in) {
entry:
  store <8 x i8> %in, <8 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v8i16_arg:
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
define void @v8i16_arg(<8 x i16> addrspace(1)* %out, <8 x i16> %in) {
entry:
  store <8 x i16> %in, <8 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v8i32_arg:
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].X
; SI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x11
; VI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x44
define void @v8i32_arg(<8 x i32> addrspace(1)* nocapture %out, <8 x i32> %in) nounwind {
entry:
  store <8 x i32> %in, <8 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v8f32_arg:
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[4].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].X
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Y
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].Z
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[5].W
; EG-DAG: T{{[0-9]\.[XYZW]}}, KC0[6].X
; SI: s_load_dwordx8 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x11
define void @v8f32_arg(<8 x float> addrspace(1)* nocapture %out, <8 x float> %in) nounwind {
entry:
  store <8 x float> %in, <8 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v16i8_arg:
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
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
define void @v16i8_arg(<16 x i8> addrspace(1)* %out, <16 x i8> %in) {
entry:
  store <16 x i8> %in, <16 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v16i16_arg:
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
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
define void @v16i16_arg(<16 x i16> addrspace(1)* %out, <16 x i16> %in) {
entry:
  store <16 x i16> %in, <16 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v16i32_arg:
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
; VI: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x64
define void @v16i32_arg(<16 x i32> addrspace(1)* nocapture %out, <16 x i32> %in) nounwind {
entry:
  store <16 x i32> %in, <16 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v16f32_arg:
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
; VI: s_load_dwordx16 s{{\[[0-9]+:[0-9]+\]}}, s[0:1], 0x64
define void @v16f32_arg(<16 x float> addrspace(1)* nocapture %out, <16 x float> %in) nounwind {
entry:
  store <16 x float> %in, <16 x float> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}kernel_arg_i64:
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2
; GCN: buffer_store_dwordx2
define void @kernel_arg_i64(i64 addrspace(1)* %out, i64 %a) nounwind {
  store i64 %a, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}f64_kernel_arg:
; SI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0x9
; SI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0xb
; VI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0x24
; VI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0x2c
; GCN: buffer_store_dwordx2
define void @f64_kernel_arg(double addrspace(1)* %out, double  %in) {
entry:
  store double %in, double addrspace(1)* %out
  ret void
}

; XFUNC-LABEL: {{^}}kernel_arg_v1i64:
; XGCN: s_load_dwordx2
; XGCN: s_load_dwordx2
; XGCN: buffer_store_dwordx2
; define void @kernel_arg_v1i64(<1 x i64> addrspace(1)* %out, <1 x i64> %a) nounwind {
;   store <1 x i64> %a, <1 x i64> addrspace(1)* %out, align 8
;   ret void
; }

; FUNC-LABEL: {{^}}i1_arg:
; SI: buffer_load_ubyte
; SI: v_and_b32_e32
; SI: buffer_store_byte
; SI: s_endpgm
define void @i1_arg(i1 addrspace(1)* %out, i1 %x) nounwind {
  store i1 %x, i1 addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_zext_i32:
; SI: buffer_load_ubyte
; SI: buffer_store_dword
; SI: s_endpgm
define void @i1_arg_zext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_zext_i64:
; SI: buffer_load_ubyte
; SI: buffer_store_dwordx2
; SI: s_endpgm
define void @i1_arg_zext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_sext_i32:
; SI: buffer_load_ubyte
; SI: buffer_store_dword
; SI: s_endpgm
define void @i1_arg_sext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
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
define void @i1_arg_sext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}
