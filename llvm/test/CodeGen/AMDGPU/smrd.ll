; RUN: llc -march=amdgcn -show-mc-encoding -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=SIVI %s
; RUN: llc -march=amdgcn -mcpu=bonaire -show-mc-encoding -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN  %s
; RUN: llc -march=amdgcn -mcpu=tonga -show-mc-encoding -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN -check-prefix=SIVI %s

; SMRD load with an immediate offset.
; GCN-LABEL: {{^}}smrd0:
; SICI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x1 ; encoding: [0x01
; VI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x4
define void @smrd0(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
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
define void @smrd1(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
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
define void @smrd2(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
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
define void @smrd3(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
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
define void @smrd4(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
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
define void @smrd5(i32 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(2)* %ptr, i64 262144
  %tmp1 = load i32, i32 addrspace(2)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; SMRD load using the load.const intrinsic with an immediate offset
; GCN-LABEL: {{^}}smrd_load_const0:
; SICI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x4 ; encoding: [0x04
; VI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x10
define amdgpu_ps void @smrd_load_const0(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg, i32 0
  %tmp20 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp
  %tmp21 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 16)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %tmp21, i1 true, i1 true) #0
  ret void
}

; SMRD load using the load.const intrinsic with the largest possible immediate
; offset.
; GCN-LABEL: {{^}}smrd_load_const1:
; SICI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xff ; encoding: [0xff
; VI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3fc
define amdgpu_ps void @smrd_load_const1(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg, i32 0
  %tmp20 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp
  %tmp21 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 1020)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %tmp21, i1 true, i1 true) #0
  ret void
}

; SMRD load using the load.const intrinsic with an offset greater than the
; largets possible immediate.
; immediate offset.
; GCN-LABEL: {{^}}smrd_load_const2:
; SI: s_movk_i32 s[[OFFSET:[0-9]]], 0x400
; SI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], s[[OFFSET]] ; encoding: [0x0[[OFFSET]]
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x100
; VI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x400
define amdgpu_ps void @smrd_load_const2(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg, i32 0
  %tmp20 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp
  %tmp21 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 1024)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %tmp21, i1 true, i1 true) #0
  ret void
}

; SMRD load with the largest possible immediate offset on VI
; GCN-LABEL: {{^}}smrd_load_const3:
; SI: s_mov_b32 [[OFFSET:s[0-9]+]], 0xffffc
; SI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3ffff
; VI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xffffc
define amdgpu_ps void @smrd_load_const3(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg, i32 0
  %tmp20 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp
  %tmp21 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 1048572)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %tmp21, i1 true, i1 true) #0
  ret void
}

; SMRD load with an offset greater than the largest possible immediate on VI
; GCN-LABEL: {{^}}smrd_load_const4:
; SIVI: s_mov_b32 [[OFFSET:s[0-9]+]], 0x100000
; SIVI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x40000
; GCN: s_endpgm
define amdgpu_ps void @smrd_load_const4(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19) #0 {
main_body:
  %tmp = getelementptr <16 x i8>, <16 x i8> addrspace(2)* %arg, i32 0
  %tmp20 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp
  %tmp21 = call float @llvm.SI.load.const(<16 x i8> %tmp20, i32 1048576)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %tmp21, i1 true, i1 true) #0
  ret void
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare float @llvm.SI.load.const(<16 x i8>, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
