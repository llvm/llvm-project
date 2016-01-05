; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=R600 --check-prefix=FUNC %s
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck --check-prefix=R600 --check-prefix=FUNC %s
; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck --check-prefix=SI-NOHSA --check-prefix=FUNC %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs | FileCheck --check-prefix=FUNC --check-prefix=CI-HSA --check-prefix=SI %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=SI-NOHSA --check-prefix=FUNC %s

;===------------------------------------------------------------------------===;
; GLOBAL ADDRESS SPACE
;===------------------------------------------------------------------------===;

; Load an i8 value from the global address space.
; FUNC-LABEL: {{^}}load_i8:
; R600: VTX_READ_8 T{{[0-9]+\.X, T[0-9]+\.X}}

; SI-NOHSA: buffer_load_ubyte v{{[0-9]+}},
; CI-HSA: flat_load_ubyte
define void @load_i8(i32 addrspace(1)* %out, i8 addrspace(1)* %in) {
  %1 = load i8, i8 addrspace(1)* %in
  %2 = zext i8 %1 to i32
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_i8_sext:
; R600: VTX_READ_8 [[DST:T[0-9]\.[XYZW]]], [[DST]]
; R600: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST]], 0.0, literal
; R600: 8
; SI-NOHSA: buffer_load_sbyte
; CI-HSA: flat_load_sbyte
define void @load_i8_sext(i32 addrspace(1)* %out, i8 addrspace(1)* %in) {
entry:
  %0 = load i8, i8 addrspace(1)* %in
  %1 = sext i8 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v2i8:
; R600: VTX_READ_8
; R600: VTX_READ_8
; SI-NOHSA: buffer_load_ubyte
; SI-NOHSA: buffer_load_ubyte
; CI-HSA: flat_load_ubyte
; CI-HSA: flat_load_ubyte
define void @load_v2i8(<2 x i32> addrspace(1)* %out, <2 x i8> addrspace(1)* %in) {
entry:
  %0 = load <2 x i8>, <2 x i8> addrspace(1)* %in
  %1 = zext <2 x i8> %0 to <2 x i32>
  store <2 x i32> %1, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v2i8_sext:
; R600-DAG: VTX_READ_8 [[DST_X:T[0-9]\.[XYZW]]], [[DST_X]]
; R600-DAG: VTX_READ_8 [[DST_Y:T[0-9]\.[XYZW]]], [[DST_Y]]
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_X]], 0.0, literal
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Y]], 0.0, literal
; R600-DAG: 8
; R600-DAG: 8

; SI-NOHSA: buffer_load_sbyte
; SI-NOHSA: buffer_load_sbyte
; CI-HSA: flat_load_sbyte
; CI-HSA: flat_load_sbyte
define void @load_v2i8_sext(<2 x i32> addrspace(1)* %out, <2 x i8> addrspace(1)* %in) {
entry:
  %0 = load <2 x i8>, <2 x i8> addrspace(1)* %in
  %1 = sext <2 x i8> %0 to <2 x i32>
  store <2 x i32> %1, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v4i8:
; R600: VTX_READ_8
; R600: VTX_READ_8
; R600: VTX_READ_8
; R600: VTX_READ_8
; SI-NOHSA: buffer_load_ubyte
; SI-NOHSA: buffer_load_ubyte
; SI-NOHSA: buffer_load_ubyte
; SI-NOHSA: buffer_load_ubyte
; CI-HSA: flat_load_ubyte
; CI-HSA: flat_load_ubyte
; CI-HSA: flat_load_ubyte
; CI-HSA: flat_load_ubyte
define void @load_v4i8(<4 x i32> addrspace(1)* %out, <4 x i8> addrspace(1)* %in) {
entry:
  %0 = load <4 x i8>, <4 x i8> addrspace(1)* %in
  %1 = zext <4 x i8> %0 to <4 x i32>
  store <4 x i32> %1, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v4i8_sext:
; R600-DAG: VTX_READ_8 [[DST_X:T[0-9]\.[XYZW]]], [[DST_X]]
; R600-DAG: VTX_READ_8 [[DST_Y:T[0-9]\.[XYZW]]], [[DST_Y]]
; R600-DAG: VTX_READ_8 [[DST_Z:T[0-9]\.[XYZW]]], [[DST_Z]]
; R600-DAG: VTX_READ_8 [[DST_W:T[0-9]\.[XYZW]]], [[DST_W]]
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_X]], 0.0, literal
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Y]], 0.0, literal
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Z]], 0.0, literal
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_W]], 0.0, literal
; R600-DAG: 8
; R600-DAG: 8
; R600-DAG: 8
; R600-DAG: 8
; SI-NOHSA: buffer_load_sbyte
; SI-NOHSA: buffer_load_sbyte
; SI-NOHSA: buffer_load_sbyte
; SI-NOHSA: buffer_load_sbyte
; CI-HSA: flat_load_sbyte
; CI-HSA: flat_load_sbyte
; CI-HSA: flat_load_sbyte
; CI-HSA: flat_load_sbyte
define void @load_v4i8_sext(<4 x i32> addrspace(1)* %out, <4 x i8> addrspace(1)* %in) {
entry:
  %0 = load <4 x i8>, <4 x i8> addrspace(1)* %in
  %1 = sext <4 x i8> %0 to <4 x i32>
  store <4 x i32> %1, <4 x i32> addrspace(1)* %out
  ret void
}

; Load an i16 value from the global address space.
; FUNC-LABEL: {{^}}load_i16:
; R600: VTX_READ_16 T{{[0-9]+\.X, T[0-9]+\.X}}
; SI-NOHSA: buffer_load_ushort
; CI-HSA: flat_load_ushort
define void @load_i16(i32 addrspace(1)* %out, i16 addrspace(1)* %in) {
entry:
  %0 = load i16	, i16	 addrspace(1)* %in
  %1 = zext i16 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_i16_sext:
; R600: VTX_READ_16 [[DST:T[0-9]\.[XYZW]]], [[DST]]
; R600: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST]], 0.0, literal
; R600: 16
; SI-NOHSA: buffer_load_sshort
; CI-HSA: flat_load_sshort
define void @load_i16_sext(i32 addrspace(1)* %out, i16 addrspace(1)* %in) {
entry:
  %0 = load i16, i16 addrspace(1)* %in
  %1 = sext i16 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v2i16:
; R600: VTX_READ_16
; R600: VTX_READ_16
; SI-NOHSA: buffer_load_ushort
; SI-NOHSA: buffer_load_ushort
; CI-HSA: flat_load_ushort
; CI-HSA: flat_load_ushort
define void @load_v2i16(<2 x i32> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) {
entry:
  %0 = load <2 x i16>, <2 x i16> addrspace(1)* %in
  %1 = zext <2 x i16> %0 to <2 x i32>
  store <2 x i32> %1, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v2i16_sext:
; R600-DAG: VTX_READ_16 [[DST_X:T[0-9]\.[XYZW]]], [[DST_X]]
; R600-DAG: VTX_READ_16 [[DST_Y:T[0-9]\.[XYZW]]], [[DST_Y]]
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_X]], 0.0, literal
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Y]], 0.0, literal
; R600-DAG: 16
; R600-DAG: 16
; SI-NOHSA: buffer_load_sshort
; SI-NOHSA: buffer_load_sshort
; CI-HSA: flat_load_sshort
; CI-HSA: flat_load_sshort
define void @load_v2i16_sext(<2 x i32> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) {
entry:
  %0 = load <2 x i16>, <2 x i16> addrspace(1)* %in
  %1 = sext <2 x i16> %0 to <2 x i32>
  store <2 x i32> %1, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v4i16:
; R600: VTX_READ_16
; R600: VTX_READ_16
; R600: VTX_READ_16
; R600: VTX_READ_16
; SI-NOHSA: buffer_load_ushort
; SI-NOHSA: buffer_load_ushort
; SI-NOHSA: buffer_load_ushort
; SI-NOHSA: buffer_load_ushort
; CI-HSA: flat_load_ushort
; CI-HSA: flat_load_ushort
; CI-HSA: flat_load_ushort
; CI-HSA: flat_load_ushort
define void @load_v4i16(<4 x i32> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) {
entry:
  %0 = load <4 x i16>, <4 x i16> addrspace(1)* %in
  %1 = zext <4 x i16> %0 to <4 x i32>
  store <4 x i32> %1, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v4i16_sext:
; R600-DAG: VTX_READ_16 [[DST_X:T[0-9]\.[XYZW]]], [[DST_X]]
; R600-DAG: VTX_READ_16 [[DST_Y:T[0-9]\.[XYZW]]], [[DST_Y]]
; R600-DAG: VTX_READ_16 [[DST_Z:T[0-9]\.[XYZW]]], [[DST_Z]]
; R600-DAG: VTX_READ_16 [[DST_W:T[0-9]\.[XYZW]]], [[DST_W]]
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_X]], 0.0, literal
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Y]], 0.0, literal
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Z]], 0.0, literal
; R600-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_W]], 0.0, literal
; R600-DAG: 16
; R600-DAG: 16
; R600-DAG: 16
; R600-DAG: 16
; SI-NOHSA: buffer_load_sshort
; SI-NOHSA: buffer_load_sshort
; SI-NOHSA: buffer_load_sshort
; SI-NOHSA: buffer_load_sshort
; CI-HSA: flat_load_sshort
; CI-HSA: flat_load_sshort
; CI-HSA: flat_load_sshort
; CI-HSA: flat_load_sshort
define void @load_v4i16_sext(<4 x i32> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) {
entry:
  %0 = load <4 x i16>, <4 x i16> addrspace(1)* %in
  %1 = sext <4 x i16> %0 to <4 x i32>
  store <4 x i32> %1, <4 x i32> addrspace(1)* %out
  ret void
}

; load an i32 value from the global address space.
; FUNC-LABEL: {{^}}load_i32:
; R600: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0

; SI-NOHSA: buffer_load_dword v{{[0-9]+}}
; CI-HSA: flat_load_dword
define void @load_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %0 = load i32, i32 addrspace(1)* %in
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; load a f32 value from the global address space.
; FUNC-LABEL: {{^}}load_f32:
; R600: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0

; SI-NOHSA: buffer_load_dword v{{[0-9]+}}
; CI-HSA: flat_load_dword
define void @load_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
entry:
  %0 = load float, float addrspace(1)* %in
  store float %0, float addrspace(1)* %out
  ret void
}

; load a v2f32 value from the global address space
; FUNC-LABEL: {{^}}load_v2f32:
; R600: MEM_RAT
; R600: VTX_READ_64
; SI-NOHSA: buffer_load_dwordx2
; CI-HSA: flat_load_dwordx2
define void @load_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %in) {
entry:
  %0 = load <2 x float>, <2 x float> addrspace(1)* %in
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_i64:
; R600: VTX_READ_64
; SI-NOHSA: buffer_load_dwordx2
; CI-HSA: flat_load_dwordx2
define void @load_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
entry:
  %0 = load i64, i64 addrspace(1)* %in
  store i64 %0, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_i64_sext:
; R600: MEM_RAT
; R600: MEM_RAT
; R600: ASHR {{[* ]*}}T{{[0-9]\.[XYZW]}}, T{{[0-9]\.[XYZW]}},  literal.x
; R600: 31
; SI-NOHSA: buffer_load_dword
; CI-HSA: flat_load_dword

define void @load_i64_sext(i64 addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %0 = load i32, i32 addrspace(1)* %in
  %1 = sext i32 %0 to i64
  store i64 %1, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_i64_zext:
; R600: MEM_RAT
; R600: MEM_RAT
define void @load_i64_zext(i64 addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %0 = load i32, i32 addrspace(1)* %in
  %1 = zext i32 %0 to i64
  store i64 %1, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v8i32:
; R600: VTX_READ_128
; R600: VTX_READ_128

; SI-NOHSA: buffer_load_dwordx4
; SI-NOHSA: buffer_load_dwordx4
; CI-HSA: flat_load_dwordx4
; CI-HSA: flat_load_dwordx4
define void @load_v8i32(<8 x i32> addrspace(1)* %out, <8 x i32> addrspace(1)* %in) {
entry:
  %0 = load <8 x i32>, <8 x i32> addrspace(1)* %in
  store <8 x i32> %0, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v16i32:
; R600: VTX_READ_128
; R600: VTX_READ_128
; R600: VTX_READ_128
; R600: VTX_READ_128

; SI-NOHSA: buffer_load_dwordx4
; SI-NOHSA: buffer_load_dwordx4
; SI-NOHSA: buffer_load_dwordx4
; SI-NOHSA: buffer_load_dwordx4
; CI-HSA: flat_load_dwordx4
; CI-HSA: flat_load_dwordx4
; CI-HSA: flat_load_dwordx4
; CI-HSA: flat_load_dwordx4
define void @load_v16i32(<16 x i32> addrspace(1)* %out, <16 x i32> addrspace(1)* %in) {
entry:
  %0 = load <16 x i32>, <16 x i32> addrspace(1)* %in
  store <16 x i32> %0, <16 x i32> addrspace(1)* %out
  ret void
}

;===------------------------------------------------------------------------===;
; CONSTANT ADDRESS SPACE
;===------------------------------------------------------------------------===;

; Load a sign-extended i8 value
; FUNC-LABEL: {{^}}load_const_i8_sext:
; R600: VTX_READ_8 [[DST:T[0-9]\.[XYZW]]], [[DST]]
; R600: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST]], 0.0, literal
; R600: 8
; SI-NOHSA: buffer_load_sbyte v{{[0-9]+}},
; CI-HSA: flat_load_sbyte v{{[0-9]+}},
define void @load_const_i8_sext(i32 addrspace(1)* %out, i8 addrspace(2)* %in) {
entry:
  %0 = load i8, i8 addrspace(2)* %in
  %1 = sext i8 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; Load an aligned i8 value
; FUNC-LABEL: {{^}}load_const_i8_aligned:
; R600: VTX_READ_8 T{{[0-9]+\.X, T[0-9]+\.X}}
; SI-NOHSA: buffer_load_ubyte v{{[0-9]+}},
; CI-HSA: flat_load_ubyte v{{[0-9]+}},
define void @load_const_i8_aligned(i32 addrspace(1)* %out, i8 addrspace(2)* %in) {
entry:
  %0 = load i8, i8 addrspace(2)* %in
  %1 = zext i8 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; Load an un-aligned i8 value
; FUNC-LABEL: {{^}}load_const_i8_unaligned:
; R600: VTX_READ_8 T{{[0-9]+\.X, T[0-9]+\.X}}
; SI-NOHSA: buffer_load_ubyte v{{[0-9]+}},
; CI-HSA: flat_load_ubyte v{{[0-9]+}},
define void @load_const_i8_unaligned(i32 addrspace(1)* %out, i8 addrspace(2)* %in) {
entry:
  %0 = getelementptr i8, i8 addrspace(2)* %in, i32 1
  %1 = load i8, i8 addrspace(2)* %0
  %2 = zext i8 %1 to i32
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; Load a sign-extended i16 value
; FUNC-LABEL: {{^}}load_const_i16_sext:
; R600: VTX_READ_16 [[DST:T[0-9]\.[XYZW]]], [[DST]]
; R600: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST]], 0.0, literal
; R600: 16
; SI-NOHSA: buffer_load_sshort
; CI-HSA: flat_load_sshort
define void @load_const_i16_sext(i32 addrspace(1)* %out, i16 addrspace(2)* %in) {
entry:
  %0 = load i16, i16 addrspace(2)* %in
  %1 = sext i16 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; Load an aligned i16 value
; FUNC-LABEL: {{^}}load_const_i16_aligned:
; R600: VTX_READ_16 T{{[0-9]+\.X, T[0-9]+\.X}}
; SI-NOHSA: buffer_load_ushort
; CI-HSA: flat_load_ushort
define void @load_const_i16_aligned(i32 addrspace(1)* %out, i16 addrspace(2)* %in) {
entry:
  %0 = load i16, i16 addrspace(2)* %in
  %1 = zext i16 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; Load an un-aligned i16 value
; FUNC-LABEL: {{^}}load_const_i16_unaligned:
; R600: VTX_READ_16 T{{[0-9]+\.X, T[0-9]+\.X}}
; SI-NOHSA: buffer_load_ushort
; CI-HSA: flat_load_ushort
define void @load_const_i16_unaligned(i32 addrspace(1)* %out, i16 addrspace(2)* %in) {
entry:
  %0 = getelementptr i16, i16 addrspace(2)* %in, i32 1
  %1 = load i16, i16 addrspace(2)* %0
  %2 = zext i16 %1 to i32
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; Load an i32 value from the constant address space.
; FUNC-LABEL: {{^}}load_const_addrspace_i32:
; R600: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0

; SI: s_load_dword s{{[0-9]+}}
define void @load_const_addrspace_i32(i32 addrspace(1)* %out, i32 addrspace(2)* %in) {
entry:
  %0 = load i32, i32 addrspace(2)* %in
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; Load a f32 value from the constant address space.
; FUNC-LABEL: {{^}}load_const_addrspace_f32:
; R600: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0

; SI: s_load_dword s{{[0-9]+}}
define void @load_const_addrspace_f32(float addrspace(1)* %out, float addrspace(2)* %in) {
  %1 = load float, float addrspace(2)* %in
  store float %1, float addrspace(1)* %out
  ret void
}

;===------------------------------------------------------------------------===;
; LOCAL ADDRESS SPACE
;===------------------------------------------------------------------------===;

; Load an i8 value from the local address space.
; FUNC-LABEL: {{^}}load_i8_local:
; R600: LDS_UBYTE_READ_RET
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_u8
define void @load_i8_local(i32 addrspace(1)* %out, i8 addrspace(3)* %in) {
  %1 = load i8, i8 addrspace(3)* %in
  %2 = zext i8 %1 to i32
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_i8_sext_local:
; R600: LDS_UBYTE_READ_RET
; R600: BFE_INT
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_i8
define void @load_i8_sext_local(i32 addrspace(1)* %out, i8 addrspace(3)* %in) {
entry:
  %0 = load i8, i8 addrspace(3)* %in
  %1 = sext i8 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v2i8_local:
; R600: LDS_UBYTE_READ_RET
; R600: LDS_UBYTE_READ_RET
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_u8
; SI: ds_read_u8
define void @load_v2i8_local(<2 x i32> addrspace(1)* %out, <2 x i8> addrspace(3)* %in) {
entry:
  %0 = load <2 x i8>, <2 x i8> addrspace(3)* %in
  %1 = zext <2 x i8> %0 to <2 x i32>
  store <2 x i32> %1, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v2i8_sext_local:
; R600-DAG: LDS_UBYTE_READ_RET
; R600-DAG: LDS_UBYTE_READ_RET
; R600-DAG: BFE_INT
; R600-DAG: BFE_INT
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_i8
; SI: ds_read_i8
define void @load_v2i8_sext_local(<2 x i32> addrspace(1)* %out, <2 x i8> addrspace(3)* %in) {
entry:
  %0 = load <2 x i8>, <2 x i8> addrspace(3)* %in
  %1 = sext <2 x i8> %0 to <2 x i32>
  store <2 x i32> %1, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v4i8_local:
; R600: LDS_UBYTE_READ_RET
; R600: LDS_UBYTE_READ_RET
; R600: LDS_UBYTE_READ_RET
; R600: LDS_UBYTE_READ_RET
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
define void @load_v4i8_local(<4 x i32> addrspace(1)* %out, <4 x i8> addrspace(3)* %in) {
entry:
  %0 = load <4 x i8>, <4 x i8> addrspace(3)* %in
  %1 = zext <4 x i8> %0 to <4 x i32>
  store <4 x i32> %1, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v4i8_sext_local:
; R600-DAG: LDS_UBYTE_READ_RET
; R600-DAG: LDS_UBYTE_READ_RET
; R600-DAG: LDS_UBYTE_READ_RET
; R600-DAG: LDS_UBYTE_READ_RET
; R600-DAG: BFE_INT
; R600-DAG: BFE_INT
; R600-DAG: BFE_INT
; R600-DAG: BFE_INT
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_i8
; SI: ds_read_i8
; SI: ds_read_i8
; SI: ds_read_i8
define void @load_v4i8_sext_local(<4 x i32> addrspace(1)* %out, <4 x i8> addrspace(3)* %in) {
entry:
  %0 = load <4 x i8>, <4 x i8> addrspace(3)* %in
  %1 = sext <4 x i8> %0 to <4 x i32>
  store <4 x i32> %1, <4 x i32> addrspace(1)* %out
  ret void
}

; Load an i16 value from the local address space.
; FUNC-LABEL: {{^}}load_i16_local:
; R600: LDS_USHORT_READ_RET
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_u16
define void @load_i16_local(i32 addrspace(1)* %out, i16 addrspace(3)* %in) {
entry:
  %0 = load i16	, i16	 addrspace(3)* %in
  %1 = zext i16 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_i16_sext_local:
; R600: LDS_USHORT_READ_RET
; R600: BFE_INT
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_i16
define void @load_i16_sext_local(i32 addrspace(1)* %out, i16 addrspace(3)* %in) {
entry:
  %0 = load i16, i16 addrspace(3)* %in
  %1 = sext i16 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v2i16_local:
; R600: LDS_USHORT_READ_RET
; R600: LDS_USHORT_READ_RET
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_u16
; SI: ds_read_u16
define void @load_v2i16_local(<2 x i32> addrspace(1)* %out, <2 x i16> addrspace(3)* %in) {
entry:
  %0 = load <2 x i16>, <2 x i16> addrspace(3)* %in
  %1 = zext <2 x i16> %0 to <2 x i32>
  store <2 x i32> %1, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v2i16_sext_local:
; R600-DAG: LDS_USHORT_READ_RET
; R600-DAG: LDS_USHORT_READ_RET
; R600-DAG: BFE_INT
; R600-DAG: BFE_INT
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_i16
; SI: ds_read_i16
define void @load_v2i16_sext_local(<2 x i32> addrspace(1)* %out, <2 x i16> addrspace(3)* %in) {
entry:
  %0 = load <2 x i16>, <2 x i16> addrspace(3)* %in
  %1 = sext <2 x i16> %0 to <2 x i32>
  store <2 x i32> %1, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v4i16_local:
; R600: LDS_USHORT_READ_RET
; R600: LDS_USHORT_READ_RET
; R600: LDS_USHORT_READ_RET
; R600: LDS_USHORT_READ_RET
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
define void @load_v4i16_local(<4 x i32> addrspace(1)* %out, <4 x i16> addrspace(3)* %in) {
entry:
  %0 = load <4 x i16>, <4 x i16> addrspace(3)* %in
  %1 = zext <4 x i16> %0 to <4 x i32>
  store <4 x i32> %1, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v4i16_sext_local:
; R600-DAG: LDS_USHORT_READ_RET
; R600-DAG: LDS_USHORT_READ_RET
; R600-DAG: LDS_USHORT_READ_RET
; R600-DAG: LDS_USHORT_READ_RET
; R600-DAG: BFE_INT
; R600-DAG: BFE_INT
; R600-DAG: BFE_INT
; R600-DAG: BFE_INT
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_i16
; SI: ds_read_i16
; SI: ds_read_i16
; SI: ds_read_i16
define void @load_v4i16_sext_local(<4 x i32> addrspace(1)* %out, <4 x i16> addrspace(3)* %in) {
entry:
  %0 = load <4 x i16>, <4 x i16> addrspace(3)* %in
  %1 = sext <4 x i16> %0 to <4 x i32>
  store <4 x i32> %1, <4 x i32> addrspace(1)* %out
  ret void
}

; load an i32 value from the local address space.
; FUNC-LABEL: {{^}}load_i32_local:
; R600: LDS_READ_RET
; SI-NOT: s_wqm_b64
; SI: s_mov_b32 m0
; SI: ds_read_b32
define void @load_i32_local(i32 addrspace(1)* %out, i32 addrspace(3)* %in) {
entry:
  %0 = load i32, i32 addrspace(3)* %in
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; load a f32 value from the local address space.
; FUNC-LABEL: {{^}}load_f32_local:
; R600: LDS_READ_RET
; SI: s_mov_b32 m0
; SI: ds_read_b32
define void @load_f32_local(float addrspace(1)* %out, float addrspace(3)* %in) {
entry:
  %0 = load float, float addrspace(3)* %in
  store float %0, float addrspace(1)* %out
  ret void
}

; load a v2f32 value from the local address space
; FUNC-LABEL: {{^}}load_v2f32_local:
; R600: LDS_READ_RET
; R600: LDS_READ_RET
; SI: s_mov_b32 m0
; SI: ds_read_b64
define void @load_v2f32_local(<2 x float> addrspace(1)* %out, <2 x float> addrspace(3)* %in) {
entry:
  %0 = load <2 x float>, <2 x float> addrspace(3)* %in
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; Test loading a i32 and v2i32 value from the same base pointer.
; FUNC-LABEL: {{^}}load_i32_v2i32_local:
; R600: LDS_READ_RET
; R600: LDS_READ_RET
; R600: LDS_READ_RET
; SI-DAG: ds_read_b32
; SI-DAG: ds_read2_b32
define void @load_i32_v2i32_local(<2 x i32> addrspace(1)* %out, i32 addrspace(3)* %in) {
  %scalar = load i32, i32 addrspace(3)* %in
  %tmp0 = bitcast i32 addrspace(3)* %in to <2 x i32> addrspace(3)*
  %vec_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(3)* %tmp0, i32 2
  %vec0 = load <2 x i32>, <2 x i32> addrspace(3)* %vec_ptr, align 4
  %vec1 = insertelement <2 x i32> <i32 0, i32 0>, i32 %scalar, i32 0
  %vec = add <2 x i32> %vec0, %vec1
  store <2 x i32> %vec, <2 x i32> addrspace(1)* %out
  ret void
}


@lds = addrspace(3) global [512 x i32] undef, align 4

; On SI we need to make sure that the base offset is a register and not
; an immediate.
; FUNC-LABEL: {{^}}load_i32_local_const_ptr:
; SI: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0
; SI: ds_read_b32 v0, v[[ZERO]] offset:4
; R600: LDS_READ_RET
define void @load_i32_local_const_ptr(i32 addrspace(1)* %out, i32 addrspace(3)* %in) {
entry:
  %tmp0 = getelementptr [512 x i32], [512 x i32] addrspace(3)* @lds, i32 0, i32 1
  %tmp1 = load i32, i32 addrspace(3)* %tmp0
  %tmp2 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  store i32 %tmp1, i32 addrspace(1)* %tmp2
  ret void
}
