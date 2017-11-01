; RUN: llc -march=amdgcn -show-mc-encoding -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=SIVI %s
; RUN: llc -march=amdgcn -mcpu=bonaire -show-mc-encoding -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN  %s
; RUN: llc -march=amdgcn -mcpu=tonga -show-mc-encoding -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN -check-prefix=SIVI %s

; SMRD load with an immediate offset.
; GCN-LABEL: {{^}}smrd0:
; SICI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x1 ; encoding: [0x01
; VI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x4
define amdgpu_kernel void @smrd0(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(2)* %ptr, i64 1
  %tmp1 = load i32, i32 addrspace(2)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with the largest possible immediate offset.
; GCN-LABEL: {{^}}smrd1:
; SICI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xff ; encoding: [0xff,0x{{[0-9]+[137]}}
; VI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3fc
define amdgpu_kernel void @smrd1(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(2)* %ptr, i64 255
  %tmp1 = load i32, i32 addrspace(2)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with an offset greater than the largest possible immediate.
; GCN-LABEL: {{^}}smrd2:
; SI: s_movk_i32 s[[OFFSET:[0-9]]], 0x400
; SI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], s[[OFFSET]] ; encoding: [0x0[[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x100
; VI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x400
; GCN: s_endpgm
define amdgpu_kernel void @smrd2(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(2)* %ptr, i64 256
  %tmp1 = load i32, i32 addrspace(2)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with a 64-bit offset
; GCN-LABEL: {{^}}smrd3:
; FIXME: There are too many copies here because we don't fold immediates
;        through REG_SEQUENCE
; SI: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[{{[0-9]:[0-9]}}], 0xb ; encoding: [0x0b
; TODO: Add VI checks
; GCN: s_endpgm
define amdgpu_kernel void @smrd3(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(2)* %ptr, i64 4294967296
  %tmp1 = load i32, i32 addrspace(2)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with the largest possible immediate offset on VI
; GCN-LABEL: {{^}}smrd4:
; SI: s_mov_b32 [[OFFSET:s[0-9]+]], 0xffffc
; SI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3ffff
; VI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xffffc
define amdgpu_kernel void @smrd4(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(2)* %ptr, i64 262143
  %tmp1 = load i32, i32 addrspace(2)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with an offset greater than the largest possible immediate on VI
; GCN-LABEL: {{^}}smrd5:
; SIVI: s_mov_b32 [[OFFSET:s[0-9]+]], 0x100000
; SIVI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x40000
; GCN: s_endpgm
define amdgpu_kernel void @smrd5(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(2)* %ptr, i64 262144
  %tmp1 = load i32, i32 addrspace(2)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_hazard:
; GCN-DAG: s_mov_b32 s3, 3
; GCN-DAG: s_mov_b32 s2, 2
; GCN-DAG: s_mov_b32 s1, 1
; GCN-DAG: s_mov_b32 s0, 0
; SI-NEXT: nop 3
; GCN-NEXT: s_buffer_load_dword s0, s[0:3], 0x0
define amdgpu_ps float @smrd_hazard(<4 x i32> inreg %desc) #0 {
main_body:
  %d0 = insertelement <4 x i32> undef, i32 0, i32 0
  %d1 = insertelement <4 x i32> %d0, i32 1, i32 1
  %d2 = insertelement <4 x i32> %d1, i32 2, i32 2
  %d3 = insertelement <4 x i32> %d2, i32 3, i32 3
  %r = call float @llvm.SI.load.const.v4i32(<4 x i32> %d3, i32 0)
  ret float %r
}

; SMRD load using the load.const.v4i32 intrinsic with an immediate offset
; GCN-LABEL: {{^}}smrd_load_const0:
; SICI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x4 ; encoding: [0x04
; VI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x10
define amdgpu_ps void @smrd_load_const0(<4 x i32> addrspace(2)* inreg %arg, <4 x i32> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(2)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(2)* %tmp
  %tmp21 = call float @llvm.SI.load.const.v4i32(<4 x i32> %tmp20, i32 16)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %tmp21, i1 true, i1 true) #0
  ret void
}

; SMRD load using the load.const.v4i32 intrinsic with the largest possible immediate
; offset.
; GCN-LABEL: {{^}}smrd_load_const1:
; SICI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xff ; encoding: [0xff
; VI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3fc
define amdgpu_ps void @smrd_load_const1(<4 x i32> addrspace(2)* inreg %arg, <4 x i32> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(2)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(2)* %tmp
  %tmp21 = call float @llvm.SI.load.const.v4i32(<4 x i32> %tmp20, i32 1020)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %tmp21, i1 true, i1 true) #0
  ret void
}

; SMRD load using the load.const.v4i32 intrinsic with an offset greater than the
; largets possible immediate.
; immediate offset.
; GCN-LABEL: {{^}}smrd_load_const2:
; SI: s_movk_i32 s[[OFFSET:[0-9]]], 0x400
; SI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], s[[OFFSET]] ; encoding: [0x0[[OFFSET]]
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x100
; VI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x400
define amdgpu_ps void @smrd_load_const2(<4 x i32> addrspace(2)* inreg %arg, <4 x i32> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(2)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(2)* %tmp
  %tmp21 = call float @llvm.SI.load.const.v4i32(<4 x i32> %tmp20, i32 1024)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %tmp21, i1 true, i1 true) #0
  ret void
}

; SMRD load with the largest possible immediate offset on VI
; GCN-LABEL: {{^}}smrd_load_const3:
; SI: s_mov_b32 [[OFFSET:s[0-9]+]], 0xffffc
; SI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3ffff
; VI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xffffc
define amdgpu_ps void @smrd_load_const3(<4 x i32> addrspace(2)* inreg %arg, <4 x i32> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(2)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(2)* %tmp
  %tmp21 = call float @llvm.SI.load.const.v4i32(<4 x i32> %tmp20, i32 1048572)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %tmp21, i1 true, i1 true) #0
  ret void
}

; SMRD load with an offset greater than the largest possible immediate on VI
; GCN-LABEL: {{^}}smrd_load_const4:
; SIVI: s_mov_b32 [[OFFSET:s[0-9]+]], 0x100000
; SIVI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x40000
; GCN: s_endpgm
define amdgpu_ps void @smrd_load_const4(<4 x i32> addrspace(2)* inreg %arg, <4 x i32> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(2)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(2)* %tmp
  %tmp21 = call float @llvm.SI.load.const.v4i32(<4 x i32> %tmp20, i32 1048576)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %tmp21, i1 true, i1 true) #0
  ret void
}

; GCN-LABEL: {{^}}smrd_sgpr_offset:
; GCN: s_buffer_load_dword s{{[0-9]}}, s[0:3], s4
define amdgpu_ps float @smrd_sgpr_offset(<4 x i32> inreg %desc, i32 inreg %offset) #0 {
main_body:
  %r = call float @llvm.SI.load.const.v4i32(<4 x i32> %desc, i32 %offset)
  ret float %r
}

; GCN-LABEL: {{^}}smrd_vgpr_offset:
; GCN: buffer_load_dword v{{[0-9]}}, v0, s[0:3], 0 offen ;
define amdgpu_ps float @smrd_vgpr_offset(<4 x i32> inreg %desc, i32 %offset) #0 {
main_body:
  %r = call float @llvm.SI.load.const.v4i32(<4 x i32> %desc, i32 %offset)
  ret float %r
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare float @llvm.SI.load.const.v4i32(<4 x i32>, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
