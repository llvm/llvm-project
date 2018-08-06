; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}s_sext_i1_to_i32:
; GCN: v_cndmask_b32_e64
; GCN: s_endpgm
define amdgpu_kernel void @s_sext_i1_to_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp eq i32 %a, %b
  %sext = sext i1 %cmp to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_s_sext_i32_to_i64:
; GCN: s_ashr_i32
; GCN: s_endpg
define amdgpu_kernel void @test_s_sext_i32_to_i64(i64 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) nounwind {
entry:
  %mul = mul i32 %a, %b
  %add = add i32 %mul, %c
  %sext = sext i32 %add to i64
  store i64 %sext, i64 addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}s_sext_i1_to_i64:
; GCN: v_cndmask_b32_e64 v[[LOREG:[0-9]+]], 0, -1, vcc
; GCN: v_mov_b32_e32 v[[HIREG:[0-9]+]], v[[LOREG]]
; GCN: buffer_store_dwordx2 v{{\[}}[[LOREG]]:[[HIREG]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @s_sext_i1_to_i64(i64 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp eq i32 %a, %b
  %sext = sext i1 %cmp to i64
  store i64 %sext, i64 addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}s_sext_i32_to_i64:
; GCN: s_ashr_i32
; GCN: s_endpgm
define amdgpu_kernel void @s_sext_i32_to_i64(i64 addrspace(1)* %out, i32 %a) nounwind {
  %sext = sext i32 %a to i64
  store i64 %sext, i64 addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}v_sext_i32_to_i64:
; GCN: v_ashr
; GCN: s_endpgm
define amdgpu_kernel void @v_sext_i32_to_i64(i64 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %sext = sext i32 %val to i64
  store i64 %sext, i64 addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}s_sext_i16_to_i64:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_bfe_i64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x100000
define amdgpu_kernel void @s_sext_i16_to_i64(i64 addrspace(1)* %out, i16 %a) nounwind {
  %sext = sext i16 %a to i64
  store i64 %sext, i64 addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}s_sext_i1_to_i16:
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1
; GCN-NEXT: buffer_store_short [[RESULT]]
define amdgpu_kernel void @s_sext_i1_to_i16(i16 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp eq i32 %a, %b
  %sext = sext i1 %cmp to i16
  store i16 %sext, i16 addrspace(1)* %out
  ret void
}

; This purpose of this test is to make sure the i16 = sign_extend i1 node
; makes it all the way throught the legalizer/optimizer to make sure
; we select this correctly.  In the s_sext_i1_to_i16, the sign_extend node
; is optimized to a select very early.
; GCN-LABEL: {{^}}s_sext_i1_to_i16_with_and:
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1
; GCN-NEXT: buffer_store_short [[RESULT]]
define amdgpu_kernel void @s_sext_i1_to_i16_with_and(i16 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
  %cmp0 = icmp eq i32 %a, %b
  %cmp1 = icmp eq i32 %c, %d
  %cmp = and i1 %cmp0, %cmp1
  %sext = sext i1 %cmp to i16
  store i16 %sext, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sext_i1_to_i16_with_and:
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1
; GCN-NEXT: buffer_store_short [[RESULT]]
define amdgpu_kernel void @v_sext_i1_to_i16_with_and(i16 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) nounwind {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %cmp0 = icmp eq i32 %a, %tid
  %cmp1 = icmp eq i32 %b, %c
  %cmp = and i1 %cmp0, %cmp1
  %sext = sext i1 %cmp to i16
  store i16 %sext, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_sext_v4i8_to_v4i32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN-DAG: s_bfe_i32 [[EXT2:s[0-9]+]], [[VAL]], 0x80010
; GCN-DAG: s_ashr_i32 [[EXT3:s[0-9]+]], [[VAL]], 24
; SI-DAG: s_bfe_i32 [[EXT1:s[0-9]+]], [[VAL]], 0x80008
; GCN-DAG: s_sext_i32_i8 [[EXT0:s[0-9]+]], [[VAL]]

; FIXME: We end up with a v_bfe instruction, because the i16 srl
; gets selected to a v_lshrrev_b16 instructions, so the input to
; the bfe is a vector registers.  To fix this we need to be able to
; optimize:
; t29: i16 = truncate t10
; t55: i16 = srl t29, Constant:i32<8>
; t63: i32 = any_extend t55
; t64: i32 = sign_extend_inreg t63, ValueType:ch:i8

; VI-DAG: v_bfe_i32 [[VEXT1:v[0-9]+]], v{{[0-9]+}}, 0, 8

; GCN-DAG: v_mov_b32_e32 [[VEXT0:v[0-9]+]], [[EXT0]]
; SI-DAG: v_mov_b32_e32 [[VEXT1:v[0-9]+]], [[EXT1]]
; GCN-DAG: v_mov_b32_e32 [[VEXT2:v[0-9]+]], [[EXT2]]
; GCN-DAG: v_mov_b32_e32 [[VEXT3:v[0-9]+]], [[EXT3]]

; GCN-DAG: buffer_store_dword [[VEXT0]]
; GCN-DAG: buffer_store_dword [[VEXT1]]
; GCN-DAG: buffer_store_dword [[VEXT2]]
; GCN-DAG: buffer_store_dword [[VEXT3]]

; GCN: s_endpgm
define amdgpu_kernel void @s_sext_v4i8_to_v4i32(i32 addrspace(1)* %out, i32 %a) nounwind {
  %cast = bitcast i32 %a to <4 x i8>
  %ext = sext <4 x i8> %cast to <4 x i32>
  %elt0 = extractelement <4 x i32> %ext, i32 0
  %elt1 = extractelement <4 x i32> %ext, i32 1
  %elt2 = extractelement <4 x i32> %ext, i32 2
  %elt3 = extractelement <4 x i32> %ext, i32 3
  store volatile i32 %elt0, i32 addrspace(1)* %out
  store volatile i32 %elt1, i32 addrspace(1)* %out
  store volatile i32 %elt2, i32 addrspace(1)* %out
  store volatile i32 %elt3, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sext_v4i8_to_v4i32:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; FIXME: need to optimize same sequence as above test to avoid
; this shift.
; VI-DAG: v_lshrrev_b16_e32 [[SH16:v[0-9]+]], 8, [[VAL]]
; GCN-DAG: v_ashrrev_i32_e32 [[EXT3:v[0-9]+]], 24, [[VAL]]
; VI-DAG: v_bfe_i32 [[EXT0:v[0-9]+]], [[VAL]], 0, 8
; VI-DAG: v_bfe_i32 [[EXT2:v[0-9]+]], [[VAL]], 16, 8
; VI-DAG: v_bfe_i32 [[EXT1:v[0-9]+]], [[SH16]], 0, 8

; SI-DAG: v_bfe_i32 [[EXT2:v[0-9]+]], [[VAL]], 16, 8
; SI-DAG: v_bfe_i32 [[EXT1:v[0-9]+]], [[VAL]], 8, 8
; SI: v_bfe_i32 [[EXT0:v[0-9]+]], [[VAL]], 0, 8

; GCN: buffer_store_dword [[EXT0]]
; GCN: buffer_store_dword [[EXT1]]
; GCN: buffer_store_dword [[EXT2]]
; GCN: buffer_store_dword [[EXT3]]
define amdgpu_kernel void @v_sext_v4i8_to_v4i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %a = load i32, i32 addrspace(1)* %in
  %cast = bitcast i32 %a to <4 x i8>
  %ext = sext <4 x i8> %cast to <4 x i32>
  %elt0 = extractelement <4 x i32> %ext, i32 0
  %elt1 = extractelement <4 x i32> %ext, i32 1
  %elt2 = extractelement <4 x i32> %ext, i32 2
  %elt3 = extractelement <4 x i32> %ext, i32 3
  store volatile i32 %elt0, i32 addrspace(1)* %out
  store volatile i32 %elt1, i32 addrspace(1)* %out
  store volatile i32 %elt2, i32 addrspace(1)* %out
  store volatile i32 %elt3, i32 addrspace(1)* %out
  ret void
}

; FIXME: s_bfe_i64, same on SI and VI
; GCN-LABEL: {{^}}s_sext_v4i16_to_v4i32:
; SI-DAG: s_ashr_i64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 48
; SI-DAG: s_ashr_i32 s{{[0-9]+}}, s{{[0-9]+}}, 16

; VI: s_ashr_i32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; VI: s_ashr_i32 s{{[0-9]+}}, s{{[0-9]+}}, 16


; GCN-DAG: s_sext_i32_i16
; GCN-DAG: s_sext_i32_i16
; GCN: s_endpgm
define amdgpu_kernel void @s_sext_v4i16_to_v4i32(i32 addrspace(1)* %out, i64 %a) nounwind {
  %cast = bitcast i64 %a to <4 x i16>
  %ext = sext <4 x i16> %cast to <4 x i32>
  %elt0 = extractelement <4 x i32> %ext, i32 0
  %elt1 = extractelement <4 x i32> %ext, i32 1
  %elt2 = extractelement <4 x i32> %ext, i32 2
  %elt3 = extractelement <4 x i32> %ext, i32 3
  store volatile i32 %elt0, i32 addrspace(1)* %out
  store volatile i32 %elt1, i32 addrspace(1)* %out
  store volatile i32 %elt2, i32 addrspace(1)* %out
  store volatile i32 %elt3, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sext_v4i16_to_v4i32:
; GCN-DAG: v_ashrrev_i32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; GCN-DAG: v_ashrrev_i32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; GCN-DAG: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 16
; GCN-DAG: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 16
; GCN: s_endpgm
define amdgpu_kernel void @v_sext_v4i16_to_v4i32(i32 addrspace(1)* %out, i64 addrspace(1)* %in) nounwind {
  %a = load i64, i64 addrspace(1)* %in
  %cast = bitcast i64 %a to <4 x i16>
  %ext = sext <4 x i16> %cast to <4 x i32>
  %elt0 = extractelement <4 x i32> %ext, i32 0
  %elt1 = extractelement <4 x i32> %ext, i32 1
  %elt2 = extractelement <4 x i32> %ext, i32 2
  %elt3 = extractelement <4 x i32> %ext, i32 3
  store volatile i32 %elt0, i32 addrspace(1)* %out
  store volatile i32 %elt1, i32 addrspace(1)* %out
  store volatile i32 %elt2, i32 addrspace(1)* %out
  store volatile i32 %elt3, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #1 = { nounwind readnone }
