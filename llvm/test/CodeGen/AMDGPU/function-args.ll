; RUN: llc -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck  -enable-var-scope -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}void_func_i1:
; GCN: v_and_b32_e32 v0, 1, v0
; GCN: buffer_store_byte v0, off
define void @void_func_i1(i1 %arg0) #0 {
  store i1 %arg0, i1 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_i1_zeroext:
; GCN: s_waitcnt
; GCN-NEXT: v_or_b32_e32 v0, 12, v0
; GCN-NOT: v0
; GCN: buffer_store_dword v0, off
define void @void_func_i1_zeroext(i1 zeroext %arg0) #0 {
  %ext = zext i1 %arg0 to i32
  %add = add i32 %ext, 12
  store i32 %add, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_i1_signext:
; GCN: s_waitcnt
; GCN-NEXT: v_add_i32_e32 v0, vcc, 12, v0
; GCN-NOT: v0
; GCN: buffer_store_dword v0, off
define void @void_func_i1_signext(i1 signext %arg0) #0 {
  %ext = sext i1 %arg0 to i32
  %add = add i32 %ext, 12
  store i32 %add, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_i8:
; GCN-NOT: v0
; GCN: buffer_store_byte v0, off
define void @void_func_i8(i8 %arg0) #0 {
  store i8 %arg0, i8 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_i8_zeroext:
; GCN-NOT: and_b32
; GCN: v_add_i32_e32 v0, vcc, 12, v0
define void @void_func_i8_zeroext(i8 zeroext %arg0) #0 {
  %ext = zext i8 %arg0 to i32
  %add = add i32 %ext, 12
  store i32 %add, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_i8_signext:
; GCN-NOT: v_bfe_i32
; GCN: v_add_i32_e32 v0, vcc, 12, v0
define void @void_func_i8_signext(i8 signext %arg0) #0 {
  %ext = sext i8 %arg0 to i32
  %add = add i32 %ext, 12
  store i32 %add, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_i16:
; GCN: buffer_store_short v0, off
define void @void_func_i16(i16 %arg0) #0 {
  store i16 %arg0, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_i16_zeroext:
; GCN-NOT: v0
; GCN: v_add_i32_e32 v0, vcc, 12, v0
define void @void_func_i16_zeroext(i16 zeroext %arg0) #0 {
  %ext = zext i16 %arg0 to i32
  %add = add i32 %ext, 12
  store i32 %add, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_i16_signext:
; GCN-NOT: v0
; GCN: v_add_i32_e32 v0, vcc, 12, v0
define void @void_func_i16_signext(i16 signext %arg0) #0 {
  %ext = sext i16 %arg0 to i32
  %add = add i32 %ext, 12
  store i32 %add, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_i32:
; GCN-NOT: v0
; GCN: buffer_store_dword v0, off
define void @void_func_i32(i32 %arg0) #0 {
  store i32 %arg0, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_i64:
; GCN-NOT: v[0:1]
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: buffer_store_dwordx2 v[0:1], off
define void @void_func_i64(i64 %arg0) #0 {
  store i64 %arg0, i64 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_f16:
; VI-NOT: v0
; CI: v_cvt_f16_f32_e32 v0, v0
; GCN: buffer_store_short v0, off
define void @void_func_f16(half %arg0) #0 {
  store half %arg0, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_f32
; GCN-NOT: v0
; GCN: buffer_store_dword v0, off
define void @void_func_f32(float %arg0) #0 {
  store float %arg0, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_f64:
; GCN-NOT: v[0:1]
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: buffer_store_dwordx2 v[0:1], off
define void @void_func_f64(double %arg0) #0 {
  store double %arg0, double addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v2i32:
; GCN-NOT: v[0:1]
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: buffer_store_dwordx2 v[0:1], off
define void @void_func_v2i32(<2 x i32> %arg0) #0 {
  store <2 x i32> %arg0, <2 x i32> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v3i32:
; GCN-DAG: buffer_store_dword v2, off
; GCN-DAG: buffer_store_dwordx2 v[0:1], off
define void @void_func_v3i32(<3 x i32> %arg0) #0 {
  store <3 x i32> %arg0, <3 x i32> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v4i32:
; GCN: buffer_store_dwordx4 v[0:3], off
define void @void_func_v4i32(<4 x i32> %arg0) #0 {
  store <4 x i32> %arg0, <4 x i32> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v5i32:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dword v4, off
define void @void_func_v5i32(<5 x i32> %arg0) #0 {
  store <5 x i32> %arg0, <5 x i32> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v8i32:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
define void @void_func_v8i32(<8 x i32> %arg0) #0 {
  store <8 x i32> %arg0, <8 x i32> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v16i32:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
; GCN-DAG: buffer_store_dwordx4 v[8:11], off
; GCN-DAG: buffer_store_dwordx4 v[12:15], off
define void @void_func_v16i32(<16 x i32> %arg0) #0 {
  store <16 x i32> %arg0, <16 x i32> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v32i32:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
; GCN-DAG: buffer_store_dwordx4 v[8:11], off
; GCN-DAG: buffer_store_dwordx4 v[12:15], off
; GCN-DAG: buffer_store_dwordx4 v[16:19], off
; GCN-DAG: buffer_store_dwordx4 v[20:23], off
; GCN-DAG: buffer_store_dwordx4 v[24:27], off
; GCN-DAG: buffer_store_dwordx4 v[28:31], off
define void @void_func_v32i32(<32 x i32> %arg0) #0 {
  store <32 x i32> %arg0, <32 x i32> addrspace(1)* undef
  ret void
}

; 1 over register limit
; GCN-LABEL: {{^}}void_func_v33i32:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
; GCN-DAG: buffer_store_dwordx4 v[8:11], off
; GCN-DAG: buffer_store_dwordx4 v[12:15], off
; GCN-DAG: buffer_load_dword [[STACKLOAD:v[0-9]+]], off, s[0:3], s5
; GCN-DAG: buffer_store_dwordx4 v[16:19], off
; GCN-DAG: buffer_store_dwordx4 v[20:23], off
; GCN-DAG: buffer_store_dwordx4 v[24:27], off
; GCN-DAG: buffer_store_dwordx4 v[28:31], off
; GCN: buffer_store_dword [[STACKLOAD]], off
define void @void_func_v33i32(<33 x i32> %arg0) #0 {
  store <33 x i32> %arg0, <33 x i32> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v2i64:
; GCN: buffer_store_dwordx4 v[0:3], off
define void @void_func_v2i64(<2 x i64> %arg0) #0 {
  store <2 x i64> %arg0, <2 x i64> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v3i64:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx2 v[4:5], off
define void @void_func_v3i64(<3 x i64> %arg0) #0 {
  store <3 x i64> %arg0, <3 x i64> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v4i64:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
define void @void_func_v4i64(<4 x i64> %arg0) #0 {
  store <4 x i64> %arg0, <4 x i64> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v5i64:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
; GCN-DAG: buffer_store_dwordx2 v[8:9], off
define void @void_func_v5i64(<5 x i64> %arg0) #0 {
  store <5 x i64> %arg0, <5 x i64> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v8i64:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
; GCN-DAG: buffer_store_dwordx4 v[8:11], off
; GCN-DAG: buffer_store_dwordx4 v[12:15], off
define void @void_func_v8i64(<8 x i64> %arg0) #0 {
  store <8 x i64> %arg0, <8 x i64> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v16i64:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
; GCN-DAG: buffer_store_dwordx4 v[8:11], off
; GCN-DAG: buffer_store_dwordx4 v[12:15], off
; GCN-DAG: buffer_store_dwordx4 v[16:19], off
; GCN-DAG: buffer_store_dwordx4 v[20:23], off
; GCN-DAG: buffer_store_dwordx4 v[24:27], off
; GCN-DAG: buffer_store_dwordx4 v[28:31], off
define void @void_func_v16i64(<16 x i64> %arg0) #0 {
  store <16 x i64> %arg0, <16 x i64> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v2i16:
; GFX9-NOT: v0
; GFX9: buffer_store_dword v0, off
define void @void_func_v2i16(<2 x i16> %arg0) #0 {
  store <2 x i16> %arg0, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v3i16:
; GCN-DAG: buffer_store_dword v0, off
; GCN-DAG: buffer_store_short v2, off
define void @void_func_v3i16(<3 x i16> %arg0) #0 {
  store <3 x i16> %arg0, <3 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v4i16:
; GFX9-NOT: v0
; GFX9-NOT: v1
; GFX9: buffer_store_dwordx2 v[0:1], off
define void @void_func_v4i16(<4 x i16> %arg0) #0 {
  store <4 x i16> %arg0, <4 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v5i16:
; GCN-DAG: buffer_store_short v4, off,
; GCN-DAG: buffer_store_dwordx2 v[1:2], off
define void @void_func_v5i16(<5 x i16> %arg0) #0 {
  store <5 x i16> %arg0, <5 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v8i16:
; GFX9-DAG: buffer_store_dwordx4 v[0:3], off
define void @void_func_v8i16(<8 x i16> %arg0) #0 {
  store <8 x i16> %arg0, <8 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v16i16:
; GFX9-DAG: buffer_store_dwordx4 v[0:3], off
; GFX9-DAG: buffer_store_dwordx4 v[4:7], off
define void @void_func_v16i16(<16 x i16> %arg0) #0 {
  store <16 x i16> %arg0, <16 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v2f32:
; GCN-NOT: v[0:1]
; GCN-NOT: v0
; GCN-NOT: v1
; GCN: buffer_store_dwordx2 v[0:1], off
define void @void_func_v2f32(<2 x float> %arg0) #0 {
  store <2 x float> %arg0, <2 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v3f32:
; GCN-DAG: buffer_store_dword v2, off
; GCN-DAG: buffer_store_dwordx2 v[0:1], off
define void @void_func_v3f32(<3 x float> %arg0) #0 {
  store <3 x float> %arg0, <3 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v4f32:
; GCN: buffer_store_dwordx4 v[0:3], off
define void @void_func_v4f32(<4 x float> %arg0) #0 {
  store <4 x float> %arg0, <4 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v8f32:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
define void @void_func_v8f32(<8 x float> %arg0) #0 {
  store <8 x float> %arg0, <8 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v16f32:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
; GCN-DAG: buffer_store_dwordx4 v[8:11], off
; GCN-DAG: buffer_store_dwordx4 v[12:15], off
define void @void_func_v16f32(<16 x float> %arg0) #0 {
  store <16 x float> %arg0, <16 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v2f64:
; GCN: buffer_store_dwordx4 v[0:3], off
define void @void_func_v2f64(<2 x double> %arg0) #0 {
  store <2 x double> %arg0, <2 x double> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v3f64:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx2 v[4:5], off
define void @void_func_v3f64(<3 x double> %arg0) #0 {
  store <3 x double> %arg0, <3 x double> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v4f64:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
define void @void_func_v4f64(<4 x double> %arg0) #0 {
  store <4 x double> %arg0, <4 x double> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v8f64:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
; GCN-DAG: buffer_store_dwordx4 v[8:11], off
; GCN-DAG: buffer_store_dwordx4 v[12:15], off
define void @void_func_v8f64(<8 x double> %arg0) #0 {
  store <8 x double> %arg0, <8 x double> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v16f64:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
; GCN-DAG: buffer_store_dwordx4 v[8:11], off
; GCN-DAG: buffer_store_dwordx4 v[12:15], off
; GCN-DAG: buffer_store_dwordx4 v[16:19], off
; GCN-DAG: buffer_store_dwordx4 v[20:23], off
; GCN-DAG: buffer_store_dwordx4 v[24:27], off
; GCN-DAG: buffer_store_dwordx4 v[28:31], off
define void @void_func_v16f64(<16 x double> %arg0) #0 {
  store <16 x double> %arg0, <16 x double> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v2f16:
; GFX9-NOT: v0
; GFX9: buffer_store_dword v0, off
define void @void_func_v2f16(<2 x half> %arg0) #0 {
  store <2 x half> %arg0, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v3f16:
; GFX9-NOT: v0
; GCN-DAG: buffer_store_dword v0, off
; GCN-DAG: buffer_store_short v2, off
define void @void_func_v3f16(<3 x half> %arg0) #0 {
  store <3 x half> %arg0, <3 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v4f16:
; GFX9-NOT: v0
; GFX9-NOT: v1
; GFX9-NOT: v[0:1]
; GFX9: buffer_store_dwordx2 v[0:1], off
define void @void_func_v4f16(<4 x half> %arg0) #0 {
  store <4 x half> %arg0, <4 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v8f16:
; GFX9-NOT: v0
; GFX9-NOT: v1
; GFX9: buffer_store_dwordx4 v[0:3], off
define void @void_func_v8f16(<8 x half> %arg0) #0 {
  store <8 x half> %arg0, <8 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v16f16:
; GFX9-NOT: v0
; GFX9-NOT: v1
; GFX9-DAG: buffer_store_dwordx4 v[0:3], off
; GFX9-DAG: buffer_store_dwordx4 v[4:7], off
define void @void_func_v16f16(<16 x half> %arg0) #0 {
  store <16 x half> %arg0, <16 x half> addrspace(1)* undef
  ret void
}

; Make sure there is no alignment requirement for passed vgprs.
; GCN-LABEL: {{^}}void_func_i32_i64_i32:
; GCN-NOT: v0
; GCN: buffer_store_dword v0, off
; GCN: buffer_store_dwordx2 v[1:2]
; GCN: buffer_store_dword v3
define void @void_func_i32_i64_i32(i32 %arg0, i64 %arg1, i32 %arg2) #0 {
  store volatile i32 %arg0, i32 addrspace(1)* undef
  store volatile i64 %arg1, i64 addrspace(1)* undef
  store volatile i32 %arg2, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_struct_i32:
; GCN-NOT: v0
; GCN: buffer_store_dword v0, off
define void @void_func_struct_i32({ i32 } %arg0) #0 {
  store { i32 } %arg0, { i32 } addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_struct_i8_i32:
; GCN-DAG: buffer_store_byte v0, off
; GCN-DAG: buffer_store_dword v1, off
define void @void_func_struct_i8_i32({ i8, i32 } %arg0) #0 {
  store { i8, i32 } %arg0, { i8, i32 } addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32:
; GCN-DAG: buffer_load_ubyte v[[ELT0:[0-9]+]], off, s[0:3], s5{{$}}
; GCN-DAG: buffer_load_dword v[[ELT1:[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; GCN-DAG: buffer_store_dword v[[ELT1]]
; GCN-DAG: buffer_store_byte v[[ELT0]]
define void @void_func_byval_struct_i8_i32({ i8, i32 }* byval %arg0) #0 {
  %arg0.load = load { i8, i32 }, { i8, i32 }* %arg0
  store { i8, i32 } %arg0.load, { i8, i32 } addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_struct_i8_i32_x2:
; GCN: buffer_load_ubyte v[[ELT0_0:[0-9]+]], off, s[0:3], s5{{$}}
; GCN: buffer_load_dword v[[ELT1_0:[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; GCN: buffer_load_ubyte v[[ELT0_1:[0-9]+]], off, s[0:3], s5 offset:8{{$}}
; GCN: buffer_load_dword v[[ELT1_1:[0-9]+]], off, s[0:3], s5 offset:12{{$}}

; GCN: ds_write_b32 v0, v0
; GCN: s_setpc_b64
define void @void_func_byval_struct_i8_i32_x2({ i8, i32 }* byval %arg0, { i8, i32 }* byval %arg1, i32 %arg2) #0 {
  %arg0.load = load volatile { i8, i32 }, { i8, i32 }* %arg0
  %arg1.load = load volatile { i8, i32 }, { i8, i32 }* %arg1
  store volatile { i8, i32 } %arg0.load, { i8, i32 } addrspace(1)* undef
  store volatile { i8, i32 } %arg1.load, { i8, i32 } addrspace(1)* undef
  store volatile i32 %arg2, i32 addrspace(3)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_byval_i32_byval_i64:
; GCN-DAG: buffer_load_dword v[[ARG0_LOAD:[0-9]+]], off, s[0:3], s5{{$}}
; GCN-DAG: buffer_load_dword v[[ARG1_LOAD0:[0-9]+]], off, s[0:3], s5 offset:8{{$}}
; GCN-DAG: buffer_load_dword v[[ARG1_LOAD1:[0-9]+]], off, s[0:3], s5 offset:12{{$}}
; GCN-DAG: buffer_store_dword v[[ARG0_LOAD]], off
; GCN-DAG: buffer_store_dwordx2 v{{\[}}[[ARG1_LOAD0]]:[[ARG1_LOAD1]]{{\]}}, off
define void @void_func_byval_i32_byval_i64(i32* byval %arg0, i64* byval %arg1) #0 {
  %arg0.load = load i32, i32* %arg0
  %arg1.load = load i64, i64* %arg1
  store i32 %arg0.load, i32 addrspace(1)* undef
  store i64 %arg1.load, i64 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v32i32_i32_i64:
; GCN-DAG: buffer_store_dwordx4 v[0:3], off
; GCN-DAG: buffer_store_dwordx4 v[4:7], off
; GCN-DAG: buffer_store_dwordx4 v[8:11], off
; GCN-DAG: buffer_store_dwordx4 v[12:15], off
; GCN-DAG: buffer_store_dwordx4 v[16:19], off
; GCN-DAG: buffer_store_dwordx4 v[20:23], off
; GCN-DAG: buffer_store_dwordx4 v[24:27], off
; GCN-DAG: buffer_store_dwordx4 v[28:31], off
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1:[0-9]+]], off, s[0:3], s5{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_0:[0-9]+]], off, s[0:3], s5 offset:4
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_1:[0-9]+]], off, s[0:3], s5 offset:8

; GCN: buffer_store_dword v[[LOAD_ARG1]]
; GCN: buffer_store_dwordx2 v{{\[}}[[LOAD_ARG2_0]]:[[LOAD_ARG2_1]]{{\]}}, off
define void @void_func_v32i32_i32_i64(<32 x i32> %arg0, i32 %arg1, i64 %arg2) #0 {
  store volatile <32 x i32> %arg0, <32 x i32> addrspace(1)* undef
  store volatile i32 %arg1, i32 addrspace(1)* undef
  store volatile i64 %arg2, i64 addrspace(1)* undef
  ret void
}

; FIXME: Different ext load types on CI vs. VI
; GCN-LABEL: {{^}}void_func_v32i32_i1_i8_i16:
; GCN-DAG: buffer_load_ubyte [[LOAD_ARG1:v[0-9]+]], off, s[0:3], s5{{$}}
; VI-DAG: buffer_load_ushort [[LOAD_ARG2:v[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; VI-DAG: buffer_load_ushort [[LOAD_ARG3:v[0-9]+]], off, s[0:3], s5 offset:8{{$}}
; VI-DAG: buffer_load_ushort [[LOAD_ARG4:v[0-9]+]], off, s[0:3], s5 offset:12{{$}}

; CI-DAG: buffer_load_dword [[LOAD_ARG2:v[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; CI-DAG: buffer_load_dword [[LOAD_ARG3:v[0-9]+]], off, s[0:3], s5 offset:8{{$}}
; CI-DAG: buffer_load_dword [[LOAD_ARG4:v[0-9]+]], off, s[0:3], s5 offset:12{{$}}

; GCN-DAG: v_and_b32_e32 [[TRUNC_ARG1_I1:v[0-9]+]], 1, [[LOAD_ARG1]]
; CI-DAG: v_cvt_f16_f32_e32 [[CVT_ARG4:v[0-9]+]], [[LOAD_ARG4]]

; GCN: buffer_store_byte [[TRUNC_ARG1_I1]], off
; GCN: buffer_store_byte [[LOAD_ARG2]], off
; GCN: buffer_store_short [[LOAD_ARG3]], off
; VI: buffer_store_short [[LOAD_ARG4]], off

; CI: buffer_store_short [[CVT_ARG4]], off
define void @void_func_v32i32_i1_i8_i16(<32 x i32> %arg0, i1 %arg1, i8 %arg2, i16 %arg3, half %arg4) #0 {
  store volatile <32 x i32> %arg0, <32 x i32> addrspace(1)* undef
  store volatile i1 %arg1, i1 addrspace(1)* undef
  store volatile i8 %arg2, i8 addrspace(1)* undef
  store volatile i16 %arg3, i16 addrspace(1)* undef
  store volatile half %arg4, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v32i32_v2i32_v2f32:
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_0:[0-9]+]], off, s[0:3], s5{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_1:[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_0:[0-9]+]], off, s[0:3], s5 offset:8{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_1:[0-9]+]], off, s[0:3], s5 offset:12{{$}}

; GCN: buffer_store_dwordx2 v{{\[}}[[LOAD_ARG1_0]]:[[LOAD_ARG1_1]]{{\]}}, off
; GCN: buffer_store_dwordx2 v{{\[}}[[LOAD_ARG2_0]]:[[LOAD_ARG2_1]]{{\]}}, off
define void @void_func_v32i32_v2i32_v2f32(<32 x i32> %arg0, <2 x i32> %arg1, <2 x float> %arg2) #0 {
  store volatile <32 x i32> %arg0, <32 x i32> addrspace(1)* undef
  store volatile <2 x i32> %arg1, <2 x i32> addrspace(1)* undef
  store volatile <2 x float> %arg2, <2 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v32i32_v2i16_v2f16:
; GFX9-DAG: buffer_load_dword [[LOAD_ARG1:v[0-9]+]], off, s[0:3], s5{{$}}
; GFX9-DAG: buffer_load_dword [[LOAD_ARG2:v[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; GFX9: buffer_store_dword [[LOAD_ARG1]], off
; GFX9: buffer_store_short [[LOAD_ARG2]], off
define void @void_func_v32i32_v2i16_v2f16(<32 x i32> %arg0, <2 x i16> %arg1, <2 x half> %arg2) #0 {
  store volatile <32 x i32> %arg0, <32 x i32> addrspace(1)* undef
  store volatile <2 x i16> %arg1, <2 x i16> addrspace(1)* undef
  store volatile <2 x half> %arg2, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v32i32_v2i64_v2f64:
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_0:[0-9]+]], off, s[0:3], s5{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_1:[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_2:[0-9]+]], off, s[0:3], s5 offset:8{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_3:[0-9]+]], off, s[0:3], s5 offset:12{{$}}

; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_0:[0-9]+]], off, s[0:3], s5 offset:16{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_1:[0-9]+]], off, s[0:3], s5 offset:20{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_2:[0-9]+]], off, s[0:3], s5 offset:24{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_3:[0-9]+]], off, s[0:3], s5 offset:28{{$}}

; GCN: buffer_store_dwordx4 v{{\[}}[[LOAD_ARG1_0]]:[[LOAD_ARG1_3]]{{\]}}, off
; GCN: buffer_store_dwordx4 v{{\[}}[[LOAD_ARG2_0]]:[[LOAD_ARG2_3]]{{\]}}, off
define void @void_func_v32i32_v2i64_v2f64(<32 x i32> %arg0, <2 x i64> %arg1, <2 x double> %arg2) #0 {
  store volatile <32 x i32> %arg0, <32 x i32> addrspace(1)* undef
  store volatile <2 x i64> %arg1, <2 x i64> addrspace(1)* undef
  store volatile <2 x double> %arg2, <2 x double> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v32i32_v4i32_v4f32:
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_0:[0-9]+]], off, s[0:3], s5{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_1:[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_2:[0-9]+]], off, s[0:3], s5 offset:8{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_3:[0-9]+]], off, s[0:3], s5 offset:12{{$}}

; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_0:[0-9]+]], off, s[0:3], s5 offset:16{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_1:[0-9]+]], off, s[0:3], s5 offset:20{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_2:[0-9]+]], off, s[0:3], s5 offset:24{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_3:[0-9]+]], off, s[0:3], s5 offset:28{{$}}

; GCN: buffer_store_dwordx4 v{{\[}}[[LOAD_ARG1_0]]:[[LOAD_ARG1_3]]{{\]}}, off
; GCN: buffer_store_dwordx4 v{{\[}}[[LOAD_ARG2_0]]:[[LOAD_ARG2_3]]{{\]}}, off
define void @void_func_v32i32_v4i32_v4f32(<32 x i32> %arg0, <4 x i32> %arg1, <4 x float> %arg2) #0 {
  store volatile <32 x i32> %arg0, <32 x i32> addrspace(1)* undef
  store volatile <4 x i32> %arg1, <4 x i32> addrspace(1)* undef
  store volatile <4 x float> %arg2, <4 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v32i32_v8i32_v8f32:
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_0:[0-9]+]], off, s[0:3], s5{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_1:[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_2:[0-9]+]], off, s[0:3], s5 offset:8{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_3:[0-9]+]], off, s[0:3], s5 offset:12{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_4:[0-9]+]], off, s[0:3], s5 offset:16{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_5:[0-9]+]], off, s[0:3], s5 offset:20{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_6:[0-9]+]], off, s[0:3], s5 offset:24{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_7:[0-9]+]], off, s[0:3], s5 offset:28{{$}}

; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_0:[0-9]+]], off, s[0:3], s5 offset:32{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_1:[0-9]+]], off, s[0:3], s5 offset:36{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_2:[0-9]+]], off, s[0:3], s5 offset:40{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_3:[0-9]+]], off, s[0:3], s5 offset:44{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_4:[0-9]+]], off, s[0:3], s5 offset:48{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_5:[0-9]+]], off, s[0:3], s5 offset:52{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_6:[0-9]+]], off, s[0:3], s5 offset:56{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_7:[0-9]+]], off, s[0:3], s5 offset:60{{$}}

; GCN: buffer_store_dwordx4 v{{\[}}[[LOAD_ARG1_4]]:[[LOAD_ARG1_7]]{{\]}}, off
; GCN: buffer_store_dwordx4 v{{\[}}[[LOAD_ARG1_0]]:[[LOAD_ARG1_3]]{{\]}}, off
; GCN: buffer_store_dwordx4 v{{\[}}[[LOAD_ARG2_4]]:[[LOAD_ARG2_7]]{{\]}}, off
; GCN: buffer_store_dwordx4 v{{\[}}[[LOAD_ARG2_0]]:[[LOAD_ARG2_3]]{{\]}}, off
define void @void_func_v32i32_v8i32_v8f32(<32 x i32> %arg0, <8 x i32> %arg1, <8 x float> %arg2) #0 {
  store volatile <32 x i32> %arg0, <32 x i32> addrspace(1)* undef
  store volatile <8 x i32> %arg1, <8 x i32> addrspace(1)* undef
  store volatile <8 x float> %arg2, <8 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}void_func_v32i32_v16i32_v16f32:
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_0:[0-9]+]], off, s[0:3], s5{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_1:[0-9]+]], off, s[0:3], s5 offset:4{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_2:[0-9]+]], off, s[0:3], s5 offset:8{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_3:[0-9]+]], off, s[0:3], s5 offset:12{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_4:[0-9]+]], off, s[0:3], s5 offset:16{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_5:[0-9]+]], off, s[0:3], s5 offset:20{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_6:[0-9]+]], off, s[0:3], s5 offset:24{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_7:[0-9]+]], off, s[0:3], s5 offset:28{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_8:[0-9]+]], off, s[0:3], s5 offset:32{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_9:[0-9]+]], off, s[0:3], s5 offset:36{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_10:[0-9]+]], off, s[0:3], s5 offset:40{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_11:[0-9]+]], off, s[0:3], s5 offset:44{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_12:[0-9]+]], off, s[0:3], s5 offset:48{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_13:[0-9]+]], off, s[0:3], s5 offset:52{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_14:[0-9]+]], off, s[0:3], s5 offset:56{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG1_15:[0-9]+]], off, s[0:3], s5 offset:60{{$}}

; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_0:[0-9]+]], off, s[0:3], s5 offset:64{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_1:[0-9]+]], off, s[0:3], s5 offset:68{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_2:[0-9]+]], off, s[0:3], s5 offset:72{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_3:[0-9]+]], off, s[0:3], s5 offset:76{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_4:[0-9]+]], off, s[0:3], s5 offset:80{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_5:[0-9]+]], off, s[0:3], s5 offset:84{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_6:[0-9]+]], off, s[0:3], s5 offset:88{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_7:[0-9]+]], off, s[0:3], s5 offset:92{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_8:[0-9]+]], off, s[0:3], s5 offset:96{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_9:[0-9]+]], off, s[0:3], s5 offset:100{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_10:[0-9]+]], off, s[0:3], s5 offset:104{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_11:[0-9]+]], off, s[0:3], s5 offset:108{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_12:[0-9]+]], off, s[0:3], s5 offset:112{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_13:[0-9]+]], off, s[0:3], s5 offset:116{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_14:[0-9]+]], off, s[0:3], s5 offset:120{{$}}
; GCN-DAG: buffer_load_dword v[[LOAD_ARG2_15:[0-9]+]], off, s[0:3], s5 offset:124{{$}}
define void @void_func_v32i32_v16i32_v16f32(<32 x i32> %arg0, <16 x i32> %arg1, <16 x float> %arg2) #0 {
  store volatile <32 x i32> %arg0, <32 x i32> addrspace(1)* undef
  store volatile <16 x i32> %arg1, <16 x i32> addrspace(1)* undef
  store volatile <16 x float> %arg2, <16 x float> addrspace(1)* undef
  ret void
}

; Check there is no crash.
; GCN-LABEL: {{^}}void_func_v16i8:
define void @void_func_v16i8(<16 x i8> %arg0) #0 {
  store volatile <16 x i8> %arg0, <16 x i8> addrspace(1)* undef
  ret void
}

; Check there is no crash.
; GCN-LABEL: {{^}}void_func_v32i32_v16i8:
define void @void_func_v32i32_v16i8(<32 x i32> %arg0, <16 x i8> %arg1) #0 {
  store volatile <32 x i32> %arg0, <32 x i32> addrspace(1)* undef
  store volatile <16 x i8> %arg1, <16 x i8> addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind }
