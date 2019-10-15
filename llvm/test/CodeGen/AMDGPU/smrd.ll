; RUN: llc -march=amdgcn -mcpu=tahiti  -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefix=SI   -check-prefix=GCN -check-prefix=SICIVI -check-prefix=SICI -check-prefix=SIVIGFX9_10 %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefix=CI   -check-prefix=GCN -check-prefix=SICIVI -check-prefix=SICI %s
; RUN: llc -march=amdgcn -mcpu=tonga   -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefix=VI   -check-prefix=GCN -check-prefix=SICIVI -check-prefix=VIGFX9_10 -check-prefix=SIVIGFX9_10 %s
; RUN: llc -march=amdgcn -mcpu=gfx900  -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefix=GFX9 -check-prefix=GCN -check-prefix=VIGFX9_10 -check-prefix=SIVIGFX9_10  %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefix=GFX10 -check-prefix=GCN -check-prefix=VIGFX9_10 -check-prefix=SIVIGFX9_10  %s

; SMRD load with an immediate offset.
; GCN-LABEL: {{^}}smrd0:
; SICI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x1 ; encoding: [0x01
; VIGFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x4
define amdgpu_kernel void @smrd0(i32 addrspace(1)* %out, i32 addrspace(4)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(4)* %ptr, i64 1
  %tmp1 = load i32, i32 addrspace(4)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with the largest possible immediate offset.
; GCN-LABEL: {{^}}smrd1:
; SICI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xff ; encoding: [0xff,0x{{[0-9]+[137]}}
; VIGFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3fc
define amdgpu_kernel void @smrd1(i32 addrspace(1)* %out, i32 addrspace(4)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(4)* %ptr, i64 255
  %tmp1 = load i32, i32 addrspace(4)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with an offset greater than the largest possible immediate.
; GCN-LABEL: {{^}}smrd2:
; SI: s_movk_i32 s[[OFFSET:[0-9]]], 0x400
; SI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], s[[OFFSET]] ; encoding: [0x0[[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x100
; VIGFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x400
; GCN: s_endpgm
define amdgpu_kernel void @smrd2(i32 addrspace(1)* %out, i32 addrspace(4)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(4)* %ptr, i64 256
  %tmp1 = load i32, i32 addrspace(4)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with a 64-bit offset
; GCN-LABEL: {{^}}smrd3:
; FIXME: There are too many copies here because we don't fold immediates
;        through REG_SEQUENCE
; SI: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[{{[0-9]:[0-9]}}], 0x13 ; encoding: [0x13
; TODO: Add VI checks
; GCN: s_endpgm
define amdgpu_kernel void @smrd3(i32 addrspace(1)* %out, [8 x i32], i32 addrspace(4)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(4)* %ptr, i64 4294967296
  %tmp1 = load i32, i32 addrspace(4)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with the largest possible immediate offset on VI
; GCN-LABEL: {{^}}smrd4:
; SI: s_mov_b32 [[OFFSET:s[0-9]+]], 0xffffc
; SI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3ffff
; VIGFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xffffc
define amdgpu_kernel void @smrd4(i32 addrspace(1)* %out, i32 addrspace(4)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(4)* %ptr, i64 262143
  %tmp1 = load i32, i32 addrspace(4)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; SMRD load with an offset greater than the largest possible immediate on VI
; GCN-LABEL: {{^}}smrd5:
; SIVIGFX9_10: s_mov_b32 [[OFFSET:s[0-9]+]], 0x100000
; SIVIGFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x40000
; GCN: s_endpgm
define amdgpu_kernel void @smrd5(i32 addrspace(1)* %out, i32 addrspace(4)* %ptr) #0 {
entry:
  %tmp = getelementptr i32, i32 addrspace(4)* %ptr, i64 262144
  %tmp1 = load i32, i32 addrspace(4)* %tmp
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_hazard:
; GCN-DAG: s_mov_b32 s3, 3
; GCN-DAG: s_mov_b32 s2, 2
; GCN-DAG: s_mov_b32 s1, 1
; GCN-DAG: s_mov_b32 s0, 0
; SI-NEXT: nop 3
; GFX10-NEXT: ; implicit-def: $vcc_hi
; GCN-NEXT: s_buffer_load_dword s0, s[0:3], 0x0
define amdgpu_ps float @smrd_hazard(<4 x i32> inreg %desc) #0 {
main_body:
  %d0 = insertelement <4 x i32> undef, i32 0, i32 0
  %d1 = insertelement <4 x i32> %d0, i32 1, i32 1
  %d2 = insertelement <4 x i32> %d1, i32 2, i32 2
  %d3 = insertelement <4 x i32> %d2, i32 3, i32 3
  %r = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %d3, i32 0, i32 0)
  ret float %r
}

; SMRD load using the load.const.v4i32 intrinsic with an immediate offset
; GCN-LABEL: {{^}}smrd_load_const0:
; SICI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x4 ; encoding: [0x04
; VIGFX9_10: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x10
define amdgpu_ps void @smrd_load_const0(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp
  %tmp21 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %tmp20, i32 16, i32 0)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %tmp21, i1 true, i1 true) #0
  ret void
}

; SMRD load using the load.const.v4i32 intrinsic with the largest possible immediate
; offset.
; GCN-LABEL: {{^}}smrd_load_const1:
; SICI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xff ; encoding: [0xff
; SICI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xff glc ; encoding: [0xff
; VIGFX9_10-DAG: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3fc ;
; VIGFX9_10-DAG: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3fc glc ;
define amdgpu_ps void @smrd_load_const1(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp
  %tmp21 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %tmp20, i32 1020, i32 0)
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %tmp22, i32 1020, i32 1)
  %s.buffer.float = bitcast i32 %s.buffer to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %s.buffer.float, i1 true, i1 true) #0
  ret void
}

; SMRD load using the load.const.v4i32 intrinsic with an offset greater than the
; largets possible immediate.
; immediate offset.
; GCN-LABEL: {{^}}smrd_load_const2:
; SI: s_movk_i32 s[[OFFSET:[0-9]]], 0x400
; SI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], s[[OFFSET]] ; encoding: [0x0[[OFFSET]]
; SI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], s[[OFFSET]] ; encoding: [0x0[[OFFSET]]
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x100
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x100
; VIGFX9_10: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x400
; VIGFX9_10: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x400
define amdgpu_ps void @smrd_load_const2(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp
  %tmp21 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %tmp20, i32 1024, i32 0)
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %tmp22, i32 1024, i32 0)
  %s.buffer.float = bitcast i32 %s.buffer to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %s.buffer.float, i1 true, i1 true) #0
  ret void
}

; SMRD load with the largest possible immediate offset on VI
; GCN-LABEL: {{^}}smrd_load_const3:
; SI: s_mov_b32 [[OFFSET:s[0-9]+]], 0xffffc
; SI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; SI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3ffff
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3ffff
; VIGFX9_10: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xffffc
; VIGFX9_10: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xffffc
define amdgpu_ps void @smrd_load_const3(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp
  %tmp21 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %tmp20, i32 1048572, i32 0)
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %tmp22, i32 1048572, i32 0)
  %s.buffer.float = bitcast i32 %s.buffer to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %s.buffer.float, i1 true, i1 true) #0
  ret void
}

; SMRD load with an offset greater than the largest possible immediate on VI
; GCN-LABEL: {{^}}smrd_load_const4:
; SIVIGFX9_10: s_mov_b32 [[OFFSET:s[0-9]+]], 0x100000
; SIVIGFX9_10: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; SIVIGFX9_10: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x40000
; CI: s_buffer_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x40000
; GCN: s_endpgm
define amdgpu_ps void @smrd_load_const4(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp
  %tmp21 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %tmp20, i32 1048576, i32 0)
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %tmp22, i32 1048576, i32 0)
  %s.buffer.float = bitcast i32 %s.buffer to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %s.buffer.float, i1 true, i1 true) #0
  ret void
}

; dwordx2 s.buffer.load
; GCN-LABEL: {{^}}s_buffer_load_dwordx2:
; VIGFX9_10: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]:[0-9]}}], 0x80
; SICI: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]:[0-9]}}], 0x20
define amdgpu_ps void @s_buffer_load_dwordx2(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in) #0 {
main_body:
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> %tmp22, i32 128, i32 0)
  %s.buffer.0 = extractelement <2 x i32> %s.buffer, i32 0
  %s.buffer.0.float = bitcast i32 %s.buffer.0 to float
  %s.buffer.1 = extractelement <2 x i32> %s.buffer, i32 1
  %s.buffer.1.float = bitcast i32 %s.buffer.1 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.0.float, float %s.buffer.1.float, float %s.buffer.0.float, float %s.buffer.1.float, i1 true, i1 true) #0
  ret void
}

; dwordx4 s.buffer.load
; GCN-LABEL: {{^}}s_buffer_load_dwordx4:
; VIGFX9_10: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]:[0-9]}}], 0x80
; SICI: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]:[0-9]}}], 0x20
define amdgpu_ps void @s_buffer_load_dwordx4(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in) #0 {
main_body:
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32> %tmp22, i32 128, i32 0)
  %s.buffer.0 = extractelement <4 x i32> %s.buffer, i32 0
  %s.buffer.0.float = bitcast i32 %s.buffer.0 to float
  %s.buffer.1 = extractelement <4 x i32> %s.buffer, i32 1
  %s.buffer.1.float = bitcast i32 %s.buffer.1 to float
  %s.buffer.2 = extractelement <4 x i32> %s.buffer, i32 2
  %s.buffer.2.float = bitcast i32 %s.buffer.2 to float
  %s.buffer.3 = extractelement <4 x i32> %s.buffer, i32 3
  %s.buffer.3.float = bitcast i32 %s.buffer.3 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.0.float, float %s.buffer.1.float, float %s.buffer.2.float, float %s.buffer.3.float, i1 true, i1 true) #0
  ret void
}

; dwordx8 s.buffer.load
; GCN-LABEL: {{^}}s_buffer_load_dwordx8:
; VIGFX9_10: s_buffer_load_dwordx8 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]:[0-9]}}], 0x80
; SICI: s_buffer_load_dwordx8 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]:[0-9]}}], 0x20
define amdgpu_ps void @s_buffer_load_dwordx8(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in) #0 {
main_body:
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call <8 x i32> @llvm.amdgcn.s.buffer.load.v8i32(<4 x i32> %tmp22, i32 128, i32 0)
  %s.buffer.0 = extractelement <8 x i32> %s.buffer, i32 0
  %s.buffer.0.float = bitcast i32 %s.buffer.0 to float
  %s.buffer.1 = extractelement <8 x i32> %s.buffer, i32 2
  %s.buffer.1.float = bitcast i32 %s.buffer.1 to float
  %s.buffer.2 = extractelement <8 x i32> %s.buffer, i32 5
  %s.buffer.2.float = bitcast i32 %s.buffer.2 to float
  %s.buffer.3 = extractelement <8 x i32> %s.buffer, i32 7
  %s.buffer.3.float = bitcast i32 %s.buffer.3 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.0.float, float %s.buffer.1.float, float %s.buffer.2.float, float %s.buffer.3.float, i1 true, i1 true) #0
  ret void
}

; dwordx16 s.buffer.load
; GCN-LABEL: {{^}}s_buffer_load_dwordx16:
; VIGFX9_10: s_buffer_load_dwordx16 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]:[0-9]}}], 0x80
; SICI: s_buffer_load_dwordx16 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]:[0-9]}}], 0x20
define amdgpu_ps void @s_buffer_load_dwordx16(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in) #0 {
main_body:
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call <16 x i32> @llvm.amdgcn.s.buffer.load.v16i32(<4 x i32> %tmp22, i32 128, i32 0)
  %s.buffer.0 = extractelement <16 x i32> %s.buffer, i32 0
  %s.buffer.0.float = bitcast i32 %s.buffer.0 to float
  %s.buffer.1 = extractelement <16 x i32> %s.buffer, i32 3
  %s.buffer.1.float = bitcast i32 %s.buffer.1 to float
  %s.buffer.2 = extractelement <16 x i32> %s.buffer, i32 12
  %s.buffer.2.float = bitcast i32 %s.buffer.2 to float
  %s.buffer.3 = extractelement <16 x i32> %s.buffer, i32 15
  %s.buffer.3.float = bitcast i32 %s.buffer.3 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.0.float, float %s.buffer.1.float, float %s.buffer.2.float, float %s.buffer.3.float, i1 true, i1 true) #0
  ret void
}

; GCN-LABEL: {{^}}smrd_sgpr_offset:
; GCN: s_buffer_load_dword s{{[0-9]}}, s[0:3], s4
define amdgpu_ps float @smrd_sgpr_offset(<4 x i32> inreg %desc, i32 inreg %offset) #0 {
main_body:
  %r = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %offset, i32 0)
  ret float %r
}

; GCN-LABEL: {{^}}smrd_vgpr_offset:
; GCN: buffer_load_dword v{{[0-9]}}, v0, s[0:3], 0 offen ;
define amdgpu_ps float @smrd_vgpr_offset(<4 x i32> inreg %desc, i32 %offset) #0 {
main_body:
  %r = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %offset, i32 0)
  ret float %r
}

; GCN-LABEL: {{^}}smrd_vgpr_offset_imm:
; GCN-NEXT: %bb.
; GCN-NEXT: buffer_load_dword v{{[0-9]}}, v0, s[0:3], 0 offen offset:4092 ;
define amdgpu_ps float @smrd_vgpr_offset_imm(<4 x i32> inreg %desc, i32 %offset) #0 {
main_body:
  %off = add i32 %offset, 4092
  %r = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %off, i32 0)
  ret float %r
}

; GCN-LABEL: {{^}}smrd_vgpr_offset_imm_too_large:
; GCN-NEXT: %bb.
; SICI-NEXT: v_add_{{i|u}}32_e32 v0, {{(vcc, )?}}0x1000, v0
; SICI-NEXT: buffer_load_dword v{{[0-9]}}, v0, s[0:3], 0 offen ;
; VIGFX9_10-NEXT: buffer_load_dword v{{[0-9]}}, v0, s[0:3], 4 offen offset:4092 ;
define amdgpu_ps float @smrd_vgpr_offset_imm_too_large(<4 x i32> inreg %desc, i32 %offset) #0 {
main_body:
  %off = add i32 %offset, 4096
  %r = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %off, i32 0)
  ret float %r
}

; GCN-LABEL: {{^}}smrd_imm_merged:
; GCN-NEXT: %bb.
; SICI-NEXT: s_buffer_load_dwordx4 s[{{[0-9]}}:{{[0-9]}}], s[0:3], 0x1
; SICI-NEXT: s_buffer_load_dwordx2 s[{{[0-9]}}:{{[0-9]}}], s[0:3], 0x7
; VIGFX9_10-NEXT: s_buffer_load_dwordx4 s[{{[0-9]}}:{{[0-9]}}], s[0:3], 0x4
; VIGFX9_10-NEXT: s_buffer_load_dwordx2 s[{{[0-9]}}:{{[0-9]}}], s[0:3], 0x1c
define amdgpu_ps void @smrd_imm_merged(<4 x i32> inreg %desc) #0 {
main_body:
  %r1 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 4, i32 0)
  %r2 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 8, i32 0)
  %r3 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 12, i32 0)
  %r4 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 16, i32 0)
  %r5 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 28, i32 0)
  %r6 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 32, i32 0)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %r1, float %r2, float %r3, float %r4, i1 true, i1 true) #0
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %r5, float %r6, float undef, float undef, i1 true, i1 true) #0
  ret void
}

; GCN-LABEL: {{^}}smrd_imm_merge_m0:
;
; GCN: s_buffer_load_dwordx2
; SICIVI: s_mov_b32 m0
; SICIVI_DAG: v_interp_p1_f32
; SICIVI_DAG: v_interp_p1_f32
; SICIVI_DAG: v_interp_p1_f32
; SICIVI_DAG: v_interp_p2_f32
; SICIVI_DAG: v_interp_p2_f32
; SICIVI_DAG: v_interp_p2_f32
;
; extractelement does not result in movrels anymore for vectors gitting 8 dwords
; SICIVI-NOT: s_mov_b32 m0
; SICIVI-NOT: v_movrels_b32_e32
; v_cndmask_b32_e32
; v_cndmask_b32_e32
;
; Merging is still thwarted on GFX9 due to s_set_gpr_idx
;
define amdgpu_ps float @smrd_imm_merge_m0(<4 x i32> inreg %desc, i32 inreg %prim, float %u, float %v) #0 {
main_body:
  %idx1.f = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 0, i32 0)
  %idx1 = bitcast float %idx1.f to i32

  %v0.x1 = call nsz float @llvm.amdgcn.interp.p1(float %u, i32 0, i32 0, i32 %prim)
  %v0.x = call nsz float @llvm.amdgcn.interp.p2(float %v0.x1, float %v, i32 0, i32 0, i32 %prim)
  %v0.y1 = call nsz float @llvm.amdgcn.interp.p1(float %u, i32 0, i32 1, i32 %prim)
  %v0.y = call nsz float @llvm.amdgcn.interp.p2(float %v0.y1, float %v, i32 0, i32 1, i32 %prim)
  %v0.z1 = call nsz float @llvm.amdgcn.interp.p1(float %u, i32 0, i32 2, i32 %prim)
  %v0.z = call nsz float @llvm.amdgcn.interp.p2(float %v0.z1, float %v, i32 0, i32 2, i32 %prim)
  %v0.tmp0 = insertelement <3 x float> undef, float %v0.x, i32 0
  %v0.tmp1 = insertelement <3 x float> %v0.tmp0, float %v0.y, i32 1
  %v0 = insertelement <3 x float> %v0.tmp1, float %v0.z, i32 2
  %a = extractelement <3 x float> %v0, i32 %idx1

  %v1.x1 = call nsz float @llvm.amdgcn.interp.p1(float %u, i32 1, i32 0, i32 %prim)
  %v1.x = call nsz float @llvm.amdgcn.interp.p2(float %v1.x1, float %v, i32 1, i32 0, i32 %prim)
  %v1.y1 = call nsz float @llvm.amdgcn.interp.p1(float %u, i32 1, i32 1, i32 %prim)
  %v1.y = call nsz float @llvm.amdgcn.interp.p2(float %v1.y1, float %v, i32 1, i32 1, i32 %prim)
  %v1.z1 = call nsz float @llvm.amdgcn.interp.p1(float %u, i32 1, i32 2, i32 %prim)
  %v1.z = call nsz float @llvm.amdgcn.interp.p2(float %v1.z1, float %v, i32 1, i32 2, i32 %prim)
  %v1.tmp0 = insertelement <3 x float> undef, float %v0.x, i32 0
  %v1.tmp1 = insertelement <3 x float> %v0.tmp0, float %v0.y, i32 1
  %v1 = insertelement <3 x float> %v0.tmp1, float %v0.z, i32 2

  %b = extractelement <3 x float> %v1, i32 %idx1
  %c = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 4, i32 0)

  %res.tmp = fadd float %a, %b
  %res = fadd float %res.tmp, %c
  ret float %res
}

; GCN-LABEL: {{^}}smrd_vgpr_merged:
; GCN-NEXT: %bb.
; GCN-NEXT: buffer_load_dwordx4 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:4
; GCN-NEXT: buffer_load_dwordx2 v[{{[0-9]}}:{{[0-9]}}], v0, s[0:3], 0 offen offset:28
define amdgpu_ps void @smrd_vgpr_merged(<4 x i32> inreg %desc, i32 %a) #0 {
main_body:
  %a1 = add i32 %a, 4
  %a2 = add i32 %a, 8
  %a3 = add i32 %a, 12
  %a4 = add i32 %a, 16
  %a5 = add i32 %a, 28
  %a6 = add i32 %a, 32
  %r1 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %a1, i32 0)
  %r2 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %a2, i32 0)
  %r3 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %a3, i32 0)
  %r4 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %a4, i32 0)
  %r5 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %a5, i32 0)
  %r6 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %a6, i32 0)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %r1, float %r2, float %r3, float %r4, i1 true, i1 true) #0
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %r5, float %r6, float undef, float undef, i1 true, i1 true) #0
  ret void
}

; GCN-LABEL: {{^}}smrd_sgpr_descriptor_promoted
; GCN: v_readfirstlane
define amdgpu_cs void @smrd_sgpr_descriptor_promoted([0 x i8] addrspace(4)* inreg noalias dereferenceable(18446744073709551615), i32) #0 {
main_body:
  %descptr = bitcast [0 x i8] addrspace(4)* %0 to <4 x i32> addrspace(4)*, !amdgpu.uniform !0
  br label %.outer_loop_header

ret_block:                                       ; preds = %.outer, %.label22, %main_body
  ret void

.outer_loop_header:
  br label %.inner_loop_header

.inner_loop_header:                                     ; preds = %.inner_loop_body, %.outer_loop_header
  %loopctr.1 = phi i32 [ 0, %.outer_loop_header ], [ %loopctr.2, %.inner_loop_body ]
  %loopctr.2 = add i32 %loopctr.1, 1
  %inner_br1 = icmp slt i32 %loopctr.2, 10
  br i1 %inner_br1, label %.inner_loop_body, label %ret_block

.inner_loop_body:
  %descriptor = load <4 x i32>, <4 x i32> addrspace(4)* %descptr, align 16, !invariant.load !0
  %load1result = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %descriptor, i32 0, i32 0)
  store float %load1result, float addrspace(1)* undef
  %inner_br2 = icmp uge i32 %1, 10
  br i1 %inner_br2, label %.inner_loop_header, label %.outer_loop_body

.outer_loop_body:
  %offset = shl i32 %loopctr.2, 6
  %load2result = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %descriptor, i32 %offset, i32 0)
  %outer_br = fcmp ueq float %load2result, 0x0
  br i1 %outer_br, label %.outer_loop_header, label %ret_block
}

; SMRD load with a non-const offset
; GCN-LABEL: {{^}}smrd_load_nonconst0:
; SIVIGFX9_10: s_buffer_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
; SIVIGFX9_10: s_buffer_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
; CI: s_buffer_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
; CI: s_buffer_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_ps void @smrd_load_nonconst0(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in, i32 inreg %ncoff) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp
  %tmp21 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %tmp20, i32 %ncoff, i32 0)
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %tmp22, i32 %ncoff, i32 0)
  %s.buffer.float = bitcast i32 %s.buffer to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %s.buffer.float, i1 true, i1 true) #0
  ret void
}

; SMRD load with a non-const non-uniform offset
; GCN-LABEL: {{^}}smrd_load_nonconst1:
; SIVIGFX9_10: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
; SIVIGFX9_10: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
; CI: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
; CI: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
; GCN: s_endpgm
define amdgpu_ps void @smrd_load_nonconst1(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in, i32 %ncoff) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp
  %tmp21 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %tmp20, i32 %ncoff, i32 0)
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %tmp22, i32 %ncoff, i32 0)
  %s.buffer.float = bitcast i32 %s.buffer to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %s.buffer.float, i1 true, i1 true) #0
  ret void
}

; SMRD load with a non-const non-uniform offset of > 4 dwords (requires splitting)
; GCN-LABEL: {{^}}smrd_load_nonconst2:
; SIVIGFX9_10: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
; SIVIGFX9_10: buffer_load_dwordx4 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
; CI: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
; CI: buffer_load_dwordx4 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen
; GCN: s_endpgm
define amdgpu_ps void @smrd_load_nonconst2(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in, i32 %ncoff) #0 {
main_body:
  %tmp = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %arg, i32 0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp
  %tmp21 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %tmp20, i32 %ncoff, i32 0)
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call <8 x i32> @llvm.amdgcn.s.buffer.load.v8i32(<4 x i32> %tmp22, i32 %ncoff, i32 0)
  %s.buffer.elt = extractelement <8 x i32> %s.buffer, i32 1
  %s.buffer.float = bitcast i32 %s.buffer.elt to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp21, float %tmp21, float %tmp21, float %s.buffer.float, i1 true, i1 true) #0
  ret void
}

; SMRD load with a non-const non-uniform offset of > 4 dwords (requires splitting)
; GCN-LABEL: {{^}}smrd_load_nonconst3:
; GCN-DAG: buffer_load_dwordx4 v[0:3], v{{[0-9]+}}, s[0:3], 0 offen ;
; GCN-DAG: buffer_load_dwordx4 v[4:7], v{{[0-9]+}}, s[0:3], 0 offen offset:16 ;
; GCN-DAG: buffer_load_dwordx4 v[8:11], v{{[0-9]+}}, s[0:3], 0 offen offset:32 ;
; GCN-DAG: buffer_load_dwordx4 v[12:15], v{{[0-9]+}}, s[0:3], 0 offen offset:48 ;
; GCN: ; return to shader part epilog
define amdgpu_ps <16 x float> @smrd_load_nonconst3(<4 x i32> inreg %rsrc, i32 %off) #0 {
main_body:
  %ld = call <16 x i32> @llvm.amdgcn.s.buffer.load.v16i32(<4 x i32> %rsrc, i32 %off, i32 0)
  %bc = bitcast <16 x i32> %ld to <16 x float>
  ret <16 x float> %bc
}

; GCN-LABEL: {{^}}smrd_load_nonconst4:
; SICI: v_add_i32_e32 v{{[0-9]+}}, vcc, 0xff8, v0 ;
; SICI-DAG: buffer_load_dwordx4 v[0:3], v{{[0-9]+}}, s[0:3], 0 offen ;
; SICI-DAG: buffer_load_dwordx4 v[4:7], v{{[0-9]+}}, s[0:3], 0 offen offset:16 ;
; SICI-DAG: buffer_load_dwordx4 v[8:11], v{{[0-9]+}}, s[0:3], 0 offen offset:32 ;
; SICI-DAG: buffer_load_dwordx4 v[12:15], v{{[0-9]+}}, s[0:3], 0 offen offset:48 ;
; VIGFX9_10-DAG: buffer_load_dwordx4 v[0:3], v{{[0-9]+}}, s[0:3], 56 offen offset:4032 ;
; VIGFX9_10-DAG: buffer_load_dwordx4 v[4:7], v{{[0-9]+}}, s[0:3], 56 offen offset:4048 ;
; VIGFX9_10-DAG: buffer_load_dwordx4 v[8:11], v{{[0-9]+}}, s[0:3], 56 offen offset:4064 ;
; VIGFX9_10-DAG: buffer_load_dwordx4 v[12:15], v{{[0-9]+}}, s[0:3], 56 offen offset:4080 ;
; GCN: ; return to shader part epilog
define amdgpu_ps <16 x float> @smrd_load_nonconst4(<4 x i32> inreg %rsrc, i32 %off) #0 {
main_body:
  %off.2 = add i32 %off, 4088
  %ld = call <16 x i32> @llvm.amdgcn.s.buffer.load.v16i32(<4 x i32> %rsrc, i32 %off.2, i32 0)
  %bc = bitcast <16 x i32> %ld to <16 x float>
  ret <16 x float> %bc
}

; GCN-LABEL: {{^}}smrd_load_nonconst5:
; SICI: v_add_i32_e32 v{{[0-9]+}}, vcc, 0x1004, v0
; SICI-DAG: buffer_load_dwordx4 v[0:3], v{{[0-9]+}}, s[0:3], 0 offen ;
; SICI-DAG: buffer_load_dwordx4 v[4:7], v{{[0-9]+}}, s[0:3], 0 offen offset:16 ;
; SICI-DAG: buffer_load_dwordx4 v[8:11], v{{[0-9]+}}, s[0:3], 0 offen offset:32 ;
; SICI-DAG: buffer_load_dwordx4 v[12:15], v{{[0-9]+}}, s[0:3], 0 offen offset:48 ;
; VIGFX9_10: s_movk_i32 s4, 0xfc0
; VIGFX9_10-DAG: buffer_load_dwordx4 v[0:3], v{{[0-9]+}}, s[0:3], s4 offen offset:68 ;
; VIGFX9_10-DAG: buffer_load_dwordx4 v[4:7], v{{[0-9]+}}, s[0:3], s4 offen offset:84 ;
; VIGFX9_10-DAG: buffer_load_dwordx4 v[8:11], v{{[0-9]+}}, s[0:3], s4 offen offset:100 ;
; VIGFX9_10-DAG: buffer_load_dwordx4 v[12:15], v{{[0-9]+}}, s[0:3], s4 offen offset:116 ;
; GCN: ; return to shader part epilog
define amdgpu_ps <16 x float> @smrd_load_nonconst5(<4 x i32> inreg %rsrc, i32 %off) #0 {
main_body:
  %off.2 = add i32 %off, 4100
  %ld = call <16 x i32> @llvm.amdgcn.s.buffer.load.v16i32(<4 x i32> %rsrc, i32 %off.2, i32 0)
  %bc = bitcast <16 x i32> %ld to <16 x float>
  ret <16 x float> %bc
}

; SMRD load dwordx2
; GCN-LABEL: {{^}}smrd_load_dwordx2:
; SIVIGFX9_10: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
; CI: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_ps void @smrd_load_dwordx2(<4 x i32> addrspace(4)* inreg %arg, <4 x i32> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <2 x i32> %arg6, <3 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, <2 x i32> %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18, float %arg19, <4 x i32> addrspace(4)* inreg %in, i32 inreg %ncoff) #0 {
main_body:
  %tmp22 = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer = call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> %tmp22, i32 %ncoff, i32 0)
  %s.buffer.float = bitcast <2 x i32> %s.buffer to <2 x float>
  %r.1 = extractelement <2 x float> %s.buffer.float, i32 0
  %r.2 = extractelement <2 x float> %s.buffer.float, i32 1
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %r.1, float %r.1, float %r.1, float %r.2, i1 true, i1 true) #0
  ret void
}

; GCN-LABEL: {{^}}smrd_uniform_loop:
;
; TODO: we should keep the loop counter in an SGPR
;
; GCN: s_buffer_load_dword
define amdgpu_ps float @smrd_uniform_loop(<4 x i32> inreg %desc, i32 %bound) #0 {
main_body:
  br label %loop

loop:
  %counter = phi i32 [ 0, %main_body ], [ %counter.next, %loop ]
  %sum = phi float [ 0.0, %main_body ], [ %sum.next, %loop ]
  %offset = shl i32 %counter, 2
  %v = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %offset, i32 0)
  %sum.next = fadd float %sum, %v
  %counter.next = add i32 %counter, 1
  %cc = icmp uge i32 %counter.next, %bound
  br i1 %cc, label %exit, label %loop

exit:
  ret float %sum.next
}


; GCN-LABEL: {{^}}smrd_uniform_loop2:
; (this test differs from smrd_uniform_loop by the more complex structure of phis,
; which used to confuse the DivergenceAnalysis after structurization)
;
; TODO: we should keep the loop counter in an SGPR and use an S_BUFFER_LOAD
;
; GCN: buffer_load_dword
define amdgpu_ps float @smrd_uniform_loop2(<4 x i32> inreg %desc, i32 %bound, i32 %bound.a) #0 {
main_body:
  br label %loop

loop:
  %counter = phi i32 [ 0, %main_body ], [ %counter.next, %loop.a ], [ %counter.next, %loop.b ]
  %sum = phi float [ 0.0, %main_body ], [ %sum.next, %loop.a ], [ %sum.next.b, %loop.b ]
  %offset = shl i32 %counter, 2
  %v = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %offset, i32 0)
  %sum.next = fadd float %sum, %v
  %counter.next = add i32 %counter, 1
  %cc = icmp uge i32 %counter.next, %bound
  br i1 %cc, label %exit, label %loop.a

loop.a:
  %cc.a = icmp uge i32 %counter.next, %bound.a
  br i1 %cc, label %loop, label %loop.b

loop.b:
  %sum.next.b = fadd float %sum.next, 1.0
  br label %loop

exit:
  ret float %sum.next
}

; This test checks that the load after some control flow with an offset based
; on a divergent shader input is correctly recognized as divergent. This was
; reduced from an actual regression. Yes, the %unused argument matters, as
; well as the fact that %arg4 is a vector.
;
; GCN-LABEL: {{^}}arg_divergence:
; GCN: buffer_load_dword v0, v0,
; GCN-NEXT: s_waitcnt
; GCN-NEXT: ; return to shader part epilog
define amdgpu_cs float @arg_divergence(i32 inreg %unused, <3 x i32> %arg4) #0 {
main_body:
  br i1 undef, label %if1, label %endif1

if1:                                              ; preds = %main_body
  store i32 0, i32 addrspace(3)* undef, align 4
  br label %endif1

endif1:                                           ; preds = %if1, %main_body
  %tmp13 = extractelement <3 x i32> %arg4, i32 0
  %tmp97 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> undef, i32 %tmp13, i32 0)
  ret float %tmp97
}

; GCN-LABEL: {{^}}s_buffer_load_f32:
; GCN: s_buffer_load_dword s0, s[0:3], s4
define amdgpu_ps void @s_buffer_load_f32(<4 x i32> inreg %rsrc, i32 inreg %offset) {
  %sgpr = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %rsrc, i32 %offset, i32 0)
  call void asm sideeffect "; use $0", "s"(float %sgpr)
  ret void
}

; GCN-LABEL: {{^}}s_buffer_load_v2f32:
; GCN: s_buffer_load_dwordx2 s[0:1], s[0:3], s4
define amdgpu_ps void @s_buffer_load_v2f32(<4 x i32> inreg %rsrc, i32 inreg %offset) {
  %sgpr = call <2 x float> @llvm.amdgcn.s.buffer.load.v2f32(<4 x i32> %rsrc, i32 %offset, i32 0)
  call void asm sideeffect "; use $0", "s"(<2 x float> %sgpr)
  ret void
}

; GCN-LABEL: {{^}}s_buffer_load_v4f32:
; GCN: s_buffer_load_dwordx4 s[0:3], s[0:3], s4
define amdgpu_ps void @s_buffer_load_v4f32(<4 x i32> inreg %rsrc, i32 inreg %offset) {
  %sgpr = call <4 x float> @llvm.amdgcn.s.buffer.load.v4f32(<4 x i32> %rsrc, i32 %offset, i32 0)
  call void asm sideeffect "; use $0", "s"(<4 x float> %sgpr)
  ret void
}

; GCN-LABEL: {{^}}s_buffer_load_v8f32:
; GCN: s_buffer_load_dwordx8 s[0:7], s[0:3], s4
define amdgpu_ps void @s_buffer_load_v8f32(<4 x i32> inreg %rsrc, i32 inreg %offset) {
  %sgpr = call <8 x float> @llvm.amdgcn.s.buffer.load.v8f32(<4 x i32> %rsrc, i32 %offset, i32 0)
  call void asm sideeffect "; use $0", "s"(<8 x float> %sgpr)
  ret void
}

; GCN-LABEL: {{^}}s_buffer_load_v16f32:
; GCN: s_buffer_load_dwordx16 s[0:15], s[0:3], s4
define amdgpu_ps void @s_buffer_load_v16f32(<4 x i32> inreg %rsrc, i32 inreg %offset) {
  %sgpr = call <16 x float> @llvm.amdgcn.s.buffer.load.v16f32(<4 x i32> %rsrc, i32 %offset, i32 0)
  call void asm sideeffect "; use $0", "s"(<16 x float> %sgpr)
  ret void
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #2
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #2

declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32>, i32, i32) #1
declare <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32>, i32, i32)
declare <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32>, i32, i32)
declare <8 x i32> @llvm.amdgcn.s.buffer.load.v8i32(<4 x i32>, i32, i32)
declare <16 x i32> @llvm.amdgcn.s.buffer.load.v16i32(<4 x i32>, i32, i32)

declare float @llvm.amdgcn.s.buffer.load.f32(<4 x i32>, i32, i32)
declare <2 x float> @llvm.amdgcn.s.buffer.load.v2f32(<4 x i32>, i32, i32)
declare <4 x float> @llvm.amdgcn.s.buffer.load.v4f32(<4 x i32>, i32, i32)
declare <8 x float> @llvm.amdgcn.s.buffer.load.v8f32(<4 x i32>, i32, i32)
declare <16 x float> @llvm.amdgcn.s.buffer.load.v16f32(<4 x i32>, i32, i32)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable }

!0 = !{}
