; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=hawaii -verify-machineinstrs < %s | FileCheck  -enable-var-scope -check-prefixes=GCN,CI %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX89 %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX89,GFX9 %s

; GCN-LABEL: {{^}}i1_func_void:
; GCN: buffer_load_ubyte v0, off
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define i1 @i1_func_void() #0 {
  %val = load i1, i1 addrspace(1)* undef
  ret i1 %val
}

; FIXME: Missing and?
; GCN-LABEL: {{^}}i1_zeroext_func_void:
; GCN: buffer_load_ubyte v0, off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define zeroext i1 @i1_zeroext_func_void() #0 {
  %val = load i1, i1 addrspace(1)* undef
  ret i1 %val
}

; GCN-LABEL: {{^}}i1_signext_func_void:
; GCN: buffer_load_ubyte v0, off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: v_bfe_i32 v0, v0, 0, 1{{$}}
; GCN-NEXT: s_setpc_b64
define signext i1 @i1_signext_func_void() #0 {
  %val = load i1, i1 addrspace(1)* undef
  ret i1 %val
}

; GCN-LABEL: {{^}}i8_func_void:
; GCN: buffer_load_ubyte v0, off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define i8 @i8_func_void() #0 {
  %val = load i8, i8 addrspace(1)* undef
  ret i8 %val
}

; GCN-LABEL: {{^}}i8_zeroext_func_void:
; GCN: buffer_load_ubyte v0, off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define zeroext i8 @i8_zeroext_func_void() #0 {
  %val = load i8, i8 addrspace(1)* undef
  ret i8 %val
}

; GCN-LABEL: {{^}}i8_signext_func_void:
; GCN: buffer_load_sbyte v0, off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define signext i8 @i8_signext_func_void() #0 {
  %val = load i8, i8 addrspace(1)* undef
  ret i8 %val
}

; GCN-LABEL: {{^}}i16_func_void:
; GCN: buffer_load_ushort v0, off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define i16 @i16_func_void() #0 {
  %val = load i16, i16 addrspace(1)* undef
  ret i16 %val
}

; GCN-LABEL: {{^}}i16_zeroext_func_void:
; GCN: buffer_load_ushort v0, off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define zeroext i16 @i16_zeroext_func_void() #0 {
  %val = load i16, i16 addrspace(1)* undef
  ret i16 %val
}

; GCN-LABEL: {{^}}i16_signext_func_void:
; GCN: buffer_load_sshort v0, off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define signext i16 @i16_signext_func_void() #0 {
  %val = load i16, i16 addrspace(1)* undef
  ret i16 %val
}

; GCN-LABEL: {{^}}i32_func_void:
; GCN: buffer_load_dword v0, off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define i32 @i32_func_void() #0 {
  %val = load i32, i32 addrspace(1)* undef
  ret i32 %val
}

; GCN-LABEL: {{^}}i64_func_void:
; GCN: buffer_load_dwordx2 v[0:1], off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define i64 @i64_func_void() #0 {
  %val = load i64, i64 addrspace(1)* undef
  ret i64 %val
}

; GCN-LABEL: {{^}}f32_func_void:
; GCN: buffer_load_dword v0, off, s[8:11], 0
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define float @f32_func_void() #0 {
  %val = load float, float addrspace(1)* undef
  ret float %val
}

; GCN-LABEL: {{^}}f64_func_void:
; GCN: buffer_load_dwordx2 v[0:1], off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define double @f64_func_void() #0 {
  %val = load double, double addrspace(1)* undef
  ret double %val
}

; GCN-LABEL: {{^}}v2f64_func_void:
; GCN: buffer_load_dwordx4 v[0:3], off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <2 x double> @v2f64_func_void() #0 {
  %val = load <2 x double>, <2 x double> addrspace(1)* undef
  ret <2 x double> %val
}

; GCN-LABEL: {{^}}v2i32_func_void:
; GCN: buffer_load_dwordx2 v[0:1], off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <2 x i32> @v2i32_func_void() #0 {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* undef
  ret <2 x i32> %val
}

; GCN-LABEL: {{^}}v3i32_func_void:
; GCN: buffer_load_dwordx4 v[0:3], off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <3 x i32> @v3i32_func_void() #0 {
  %val = load <3 x i32>, <3 x i32> addrspace(1)* undef
  ret <3 x i32> %val
}

; GCN-LABEL: {{^}}v4i32_func_void:
; GCN: buffer_load_dwordx4 v[0:3], off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <4 x i32> @v4i32_func_void() #0 {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* undef
  ret <4 x i32> %val
}

; GCN-LABEL: {{^}}v5i32_func_void:
; GCN-DAG: buffer_load_dword v4, off
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <5 x i32> @v5i32_func_void() #0 {
  %val = load volatile <5 x i32>, <5 x i32> addrspace(1)* undef
  ret <5 x i32> %val
}

; GCN-LABEL: {{^}}v8i32_func_void:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <8 x i32> @v8i32_func_void() #0 {
  %ptr = load volatile <8 x i32> addrspace(1)*, <8 x i32> addrspace(1)* addrspace(4)* undef
  %val = load <8 x i32>, <8 x i32> addrspace(1)* %ptr
  ret <8 x i32> %val
}

; GCN-LABEL: {{^}}v16i32_func_void:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN-DAG: buffer_load_dwordx4 v[8:11], off
; GCN-DAG: buffer_load_dwordx4 v[12:15], off
; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <16 x i32> @v16i32_func_void() #0 {
  %ptr = load volatile <16 x i32> addrspace(1)*, <16 x i32> addrspace(1)* addrspace(4)* undef
  %val = load <16 x i32>, <16 x i32> addrspace(1)* %ptr
  ret <16 x i32> %val
}

; GCN-LABEL: {{^}}v32i32_func_void:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN-DAG: buffer_load_dwordx4 v[8:11], off
; GCN-DAG: buffer_load_dwordx4 v[12:15], off
; GCN-DAG: buffer_load_dwordx4 v[16:19], off
; GCN-DAG: buffer_load_dwordx4 v[20:23], off
; GCN-DAG: buffer_load_dwordx4 v[24:27], off
; GCN-DAG: buffer_load_dwordx4 v[28:31], off
; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <32 x i32> @v32i32_func_void() #0 {
  %ptr = load volatile <32 x i32> addrspace(1)*, <32 x i32> addrspace(1)* addrspace(4)* undef
  %val = load <32 x i32>, <32 x i32> addrspace(1)* %ptr
  ret <32 x i32> %val
}

; GCN-LABEL: {{^}}v2i64_func_void:
; GCN: buffer_load_dwordx4 v[0:3], off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <2 x i64> @v2i64_func_void() #0 {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* undef
  ret <2 x i64> %val
}

; GCN-LABEL: {{^}}v3i64_func_void:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <3 x i64> @v3i64_func_void() #0 {
  %ptr = load volatile <3 x i64> addrspace(1)*, <3 x i64> addrspace(1)* addrspace(4)* undef
  %val = load <3 x i64>, <3 x i64> addrspace(1)* %ptr
  ret <3 x i64> %val
}

; GCN-LABEL: {{^}}v4i64_func_void:
; GCN: buffer_load_dwordx4 v[0:3], off
; GCN: buffer_load_dwordx4 v[4:7], off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <4 x i64> @v4i64_func_void() #0 {
  %ptr = load volatile <4 x i64> addrspace(1)*, <4 x i64> addrspace(1)* addrspace(4)* undef
  %val = load <4 x i64>, <4 x i64> addrspace(1)* %ptr
  ret <4 x i64> %val
}

; GCN-LABEL: {{^}}v5i64_func_void:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN-DAG: buffer_load_dwordx4 v[8:11], off
; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <5 x i64> @v5i64_func_void() #0 {
  %ptr = load volatile <5 x i64> addrspace(1)*, <5 x i64> addrspace(1)* addrspace(4)* undef
  %val = load <5 x i64>, <5 x i64> addrspace(1)* %ptr
  ret <5 x i64> %val
}

; GCN-LABEL: {{^}}v8i64_func_void:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN-DAG: buffer_load_dwordx4 v[8:11], off
; GCN-DAG: buffer_load_dwordx4 v[12:15], off
; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <8 x i64> @v8i64_func_void() #0 {
  %ptr = load volatile <8 x i64> addrspace(1)*, <8 x i64> addrspace(1)* addrspace(4)* undef
  %val = load <8 x i64>, <8 x i64> addrspace(1)* %ptr
  ret <8 x i64> %val
}

; GCN-LABEL: {{^}}v16i64_func_void:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN-DAG: buffer_load_dwordx4 v[8:11], off
; GCN-DAG: buffer_load_dwordx4 v[12:15], off
; GCN-DAG: buffer_load_dwordx4 v[16:19], off
; GCN-DAG: buffer_load_dwordx4 v[20:23], off
; GCN-DAG: buffer_load_dwordx4 v[24:27], off
; GCN-DAG: buffer_load_dwordx4 v[28:31], off
; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define <16 x i64> @v16i64_func_void() #0 {
  %ptr = load volatile <16 x i64> addrspace(1)*, <16 x i64> addrspace(1)* addrspace(4)* undef
  %val = load <16 x i64>, <16 x i64> addrspace(1)* %ptr
  ret <16 x i64> %val
}

; GCN-LABEL: {{^}}v2i16_func_void:
; GFX9: buffer_load_dword v0, off
; GFX9-NEXT: s_waitcnt vmcnt(0)
; GFX9-NEXT: s_setpc_b64
define <2 x i16> @v2i16_func_void() #0 {
  %val = load <2 x i16>, <2 x i16> addrspace(1)* undef
  ret <2 x i16> %val
}

; GCN-LABEL: {{^}}v3i16_func_void:
; GFX9: buffer_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, off
; GFX9-NEXT: s_waitcnt vmcnt(0)
; GFX9-NEXT: s_setpc_b64
define <3 x i16> @v3i16_func_void() #0 {
  %val = load <3 x i16>, <3 x i16> addrspace(1)* undef
  ret <3 x i16> %val
}

; GCN-LABEL: {{^}}v4i16_func_void:
; GFX9: buffer_load_dwordx2 v[0:1], off
; GFX9-NEXT: s_waitcnt vmcnt(0)
; GFX9-NEXT: s_setpc_b64
define <4 x i16> @v4i16_func_void() #0 {
  %val = load <4 x i16>, <4 x i16> addrspace(1)* undef
  ret <4 x i16> %val
}

; GCN-LABEL: {{^}}v4f16_func_void:
; GFX9: buffer_load_dwordx2 v[0:1], off
; GFX9-NEXT: s_waitcnt vmcnt(0)
; GFX9-NEXT: s_setpc_b64
define <4 x half> @v4f16_func_void() #0 {
  %val = load <4 x half>, <4 x half> addrspace(1)* undef
  ret <4 x half> %val
}

; FIXME: Should not scalarize
; GCN-LABEL: {{^}}v5i16_func_void:
; GFX9: buffer_load_dwordx2 v[0:1]
; GFX9: buffer_load_ushort v4
; GFX9: v_lshrrev_b32_e32 v5, 16, v0
; GFX9: v_lshrrev_b32_e32 v3, 16, v1
; GCN: s_setpc_b64
define <5 x i16> @v5i16_func_void() #0 {
  %ptr = load volatile <5 x i16> addrspace(1)*, <5 x i16> addrspace(1)* addrspace(4)* undef
  %val = load <5 x i16>, <5 x i16> addrspace(1)* %ptr
  ret <5 x i16> %val
}

; GCN-LABEL: {{^}}v8i16_func_void:
; GFX9-DAG: buffer_load_dwordx4 v[0:3], off
; GFX9: s_waitcnt vmcnt(0)
; GFX9-NEXT: s_setpc_b64
define <8 x i16> @v8i16_func_void() #0 {
  %ptr = load volatile <8 x i16> addrspace(1)*, <8 x i16> addrspace(1)* addrspace(4)* undef
  %val = load <8 x i16>, <8 x i16> addrspace(1)* %ptr
  ret <8 x i16> %val
}

; GCN-LABEL: {{^}}v16i16_func_void:
; GFX9: buffer_load_dwordx4 v[0:3], off
; GFX9: buffer_load_dwordx4 v[4:7], off
; GFX9: s_waitcnt vmcnt(0)
; GFX9-NEXT: s_setpc_b64
define <16 x i16> @v16i16_func_void() #0 {
  %ptr = load volatile <16 x i16> addrspace(1)*, <16 x i16> addrspace(1)* addrspace(4)* undef
  %val = load <16 x i16>, <16 x i16> addrspace(1)* %ptr
  ret <16 x i16> %val
}

; FIXME: Should pack
; GCN-LABEL: {{^}}v16i8_func_void:
; GCN-DAG: v12
; GCN-DAG: v13
; GCN-DAG: v14
; GCN-DAG: v15
define <16 x i8> @v16i8_func_void() #0 {
  %ptr = load volatile <16 x i8> addrspace(1)*, <16 x i8> addrspace(1)* addrspace(4)* undef
  %val = load <16 x i8>, <16 x i8> addrspace(1)* %ptr
  ret <16 x i8> %val
}

; FIXME: Should pack
; GCN-LABEL: {{^}}v4i8_func_void:
; GCN: buffer_load_dword v0
; GCN-DAG: v_lshrrev_b32_e32 v1, 8, v0
; GCN-DAG: v_lshrrev_b32_e32 v2, 16, v0
; GCN-DAG: v_lshrrev_b32_e32 v3, 24, v0
; GCN: s_setpc_b64
define <4  x i8> @v4i8_func_void() #0 {
  %ptr = load volatile <4  x i8> addrspace(1)*, <4  x i8> addrspace(1)* addrspace(4)* undef
  %val = load <4  x i8>, <4  x i8> addrspace(1)* %ptr
  ret <4  x i8> %val
}

; GCN-LABEL: {{^}}struct_i8_i32_func_void:
; GCN-DAG: buffer_load_dword v1
; GCN-DAG: buffer_load_ubyte v0
; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define {i8, i32} @struct_i8_i32_func_void() #0 {
  %val = load { i8, i32 }, { i8, i32 } addrspace(1)* undef
  ret { i8, i32 } %val
}

; GCN-LABEL: {{^}}void_func_sret_struct_i8_i32:
; GCN: buffer_load_ubyte [[VAL0:v[0-9]+]]
; GCN: buffer_load_dword [[VAL1:v[0-9]+]]
; GCN: buffer_store_byte [[VAL0]], v0, s[0:3], s4 offen{{$}}
; GCN: buffer_store_dword [[VAL1]], v0, s[0:3], s4 offen offset:4{{$}}
define void @void_func_sret_struct_i8_i32({ i8, i32 } addrspace(5)* sret %arg0) #0 {
  %val0 = load volatile i8, i8 addrspace(1)* undef
  %val1 = load volatile i32, i32 addrspace(1)* undef
  %gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %arg0, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %arg0, i32 0, i32 1
  store i8 %val0, i8 addrspace(5)* %gep0
  store i32 %val1, i32 addrspace(5)* %gep1
  ret void
}

; FIXME: Should be able to fold offsets in all of these pre-gfx9. Call
; lowering introduces an extra CopyToReg/CopyFromReg obscuring the
; AssertZext inserted. Not using it introduces the spills.

; GCN-LABEL: {{^}}v33i32_func_void:
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:4{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:8{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:12{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:16{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:20{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:24{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:28{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:32{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:36{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:40{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:44{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:48{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:52{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:56{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:60{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:64{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:68{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:72{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:76{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:80{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:84{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:88{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:92{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:96{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:100{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:104{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:108{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:112{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:116{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:120{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:124{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:128{{$}}
; GFX9: s_waitcnt vmcnt(0)
; GFX9-NEXT: s_setpc_b64
define <33 x i32> @v33i32_func_void() #0 {
  %ptr = load volatile <33 x i32> addrspace(1)*, <33 x i32> addrspace(1)* addrspace(4)* undef
  %val = load <33 x i32>, <33 x i32> addrspace(1)* %ptr
  ret <33 x i32> %val
}

; GCN-LABEL: {{^}}struct_v32i32_i32_func_void:
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:4{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:8{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:12{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:16{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:20{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:24{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:28{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:32{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:36{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:40{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:44{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:48{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:52{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:56{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:60{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:64{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:68{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:72{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:76{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:80{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:84{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:88{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:92{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:96{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:100{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:104{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:108{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:112{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:116{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:120{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:124{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:128{{$}}
; GFX9: s_waitcnt vmcnt(0)
; GFX9-NEXT: s_setpc_b64
define { <32 x i32>, i32 } @struct_v32i32_i32_func_void() #0 {
  %ptr = load volatile { <32 x i32>, i32 } addrspace(1)*, { <32 x i32>, i32 } addrspace(1)* addrspace(4)* undef
  %val = load { <32 x i32>, i32 }, { <32 x i32>, i32 } addrspace(1)* %ptr
  ret { <32 x i32>, i32 }%val
}

; GCN-LABEL: {{^}}struct_i32_v32i32_func_void:
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:128{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:132{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:136{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:140{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:144{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:148{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:152{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:156{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:160{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:164{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:168{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:172{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:176{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:180{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:184{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:188{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:192{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:196{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:200{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:204{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:208{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:212{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:216{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:220{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:224{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:228{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:232{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:236{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:240{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:244{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:248{{$}}
; GFX9-DAG: buffer_store_dword v{{[0-9]+}}, v0, s[0:3], s4 offen offset:252{{$}}
; GFX9: s_waitcnt vmcnt(0)
; GFX9-NEXT: s_setpc_b64
define { i32, <32 x i32> } @struct_i32_v32i32_func_void() #0 {
  %ptr = load volatile { i32, <32 x i32> } addrspace(1)*, { i32, <32 x i32> } addrspace(1)* addrspace(4)* undef
  %val = load { i32, <32 x i32> }, { i32, <32 x i32> } addrspace(1)* %ptr
  ret { i32, <32 x i32> }%val
}

attributes #0 = { nounwind }
