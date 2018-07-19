; RUN: llc -march=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-sdwa-peephole=0 < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.y() nounwind readnone

; FUNC-LABEL: {{^}}test_copy_v4i8:
; GCN: {{buffer|flat}}_load_dword [[REG:v[0-9]+]]
; GCN: buffer_store_dword [[REG]]
; GCN: s_endpgm
define amdgpu_kernel void @test_copy_v4i8(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %in) nounwind {
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %in, i32 %tid.x
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %gep, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_x2:
; GCN: {{buffer|flat}}_load_dword [[REG:v[0-9]+]]
; GCN: buffer_store_dword [[REG]]
; GCN: buffer_store_dword [[REG]]
; GCN: s_endpgm
define amdgpu_kernel void @test_copy_v4i8_x2(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %in) nounwind {
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %in, i32 %tid.x
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %gep, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out1, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_x3:
; GCN: {{buffer|flat}}_load_dword [[REG:v[0-9]+]]
; GCN: buffer_store_dword [[REG]]
; GCN: buffer_store_dword [[REG]]
; GCN: buffer_store_dword [[REG]]
; GCN: s_endpgm
define amdgpu_kernel void @test_copy_v4i8_x3(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %out2, <4 x i8> addrspace(1)* %in) nounwind {
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %in, i32 %tid.x
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %gep, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out1, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out2, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_x4:
; GCN: {{buffer|flat}}_load_dword [[REG:v[0-9]+]]
; GCN: buffer_store_dword [[REG]]
; GCN: buffer_store_dword [[REG]]
; GCN: buffer_store_dword [[REG]]
; GCN: buffer_store_dword [[REG]]
; GCN: s_endpgm
define amdgpu_kernel void @test_copy_v4i8_x4(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %out2, <4 x i8> addrspace(1)* %out3, <4 x i8> addrspace(1)* %in) nounwind {
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %in, i32 %tid.x
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %gep, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out1, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out2, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out3, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_extra_use:
; GCN: {{buffer|flat}}_load_dword
; GCN-DAG: v_lshrrev_b32
; GCN: v_and_b32
; GCN: v_or_b32
; GCN-DAG: buffer_store_dword
; GCN-DAG: buffer_store_dword

; GCN: s_endpgm
define amdgpu_kernel void @test_copy_v4i8_extra_use(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %in) nounwind {
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %in, i32 %tid.x
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %gep, align 4
  %add = add <4 x i8> %val, <i8 9, i8 9, i8 9, i8 9>
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %add, <4 x i8> addrspace(1)* %out1, align 4
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; FUNC-LABEL: {{^}}test_copy_v4i8_x2_extra_use:
; GCN: {{buffer|flat}}_load_dword
; GCN-DAG: v_lshrrev_b32
; SI-DAG: v_add_i32
; VI-DAG: v_add_u16
; GCN-DAG: v_and_b32
; GCN-DAG: v_or_b32
; GCN-DAG: {{buffer|flat}}_store_dword
; GCN: {{buffer|flat}}_store_dword
; GCN: {{buffer|flat}}_store_dword
; GCN: s_endpgm
define amdgpu_kernel void @test_copy_v4i8_x2_extra_use(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %out2, <4 x i8> addrspace(1)* %in) nounwind {
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %in.ptr = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %in, i32 %tid.x
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %in.ptr, align 4
  %add = add <4 x i8> %val, <i8 9, i8 9, i8 9, i8 9>
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %add, <4 x i8> addrspace(1)* %out1, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out2, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v3i8_align4:
; GCN: {{buffer|flat}}_load_dword
; GCN-DAG: buffer_store_short v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN-DAG: buffer_store_byte v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:2{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @test_copy_v3i8_align4(<3 x i8> addrspace(1)* %out, <3 x i8> addrspace(1)* %in) nounwind {
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <3 x i8>, <3 x i8> addrspace(1)* %in, i32 %tid.x
  %val = load <3 x i8>, <3 x i8> addrspace(1)* %gep, align 4
  store <3 x i8> %val, <3 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v3i8_align2:
; GCN-DAG: {{buffer|flat}}_load_ushort v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN-DAG: {{buffer|flat}}_load_ubyte v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:2{{$}}
; GCN-DAG: buffer_store_short v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN-DAG: buffer_store_byte v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:2{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @test_copy_v3i8_align2(<3 x i8> addrspace(1)* %out, <3 x i8> addrspace(1)* %in) nounwind {
  %val = load <3 x i8>, <3 x i8> addrspace(1)* %in, align 2
  store <3 x i8> %val, <3 x i8> addrspace(1)* %out, align 2
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v3i8_align1:
; GCN: {{buffer|flat}}_load_ubyte
; GCN: {{buffer|flat}}_load_ubyte
; GCN: {{buffer|flat}}_load_ubyte

; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: s_endpgm
define amdgpu_kernel void @test_copy_v3i8_align1(<3 x i8> addrspace(1)* %out, <3 x i8> addrspace(1)* %in) nounwind {
  %val = load <3 x i8>, <3 x i8> addrspace(1)* %in, align 1
  store <3 x i8> %val, <3 x i8> addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_volatile_load:
; GCN: {{buffer|flat}}_load_ubyte
; GCN: {{buffer|flat}}_load_ubyte
; GCN: {{buffer|flat}}_load_ubyte
; GCN: {{buffer|flat}}_load_ubyte
; GCN: buffer_store_dword
; GCN: s_endpgm
define amdgpu_kernel void @test_copy_v4i8_volatile_load(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load volatile <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_copy_v4i8_volatile_store:
; GCN: {{buffer|flat}}_load_ubyte
; GCN: {{buffer|flat}}_load_ubyte
; GCN: {{buffer|flat}}_load_ubyte
; GCN: {{buffer|flat}}_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: s_endpgm
define amdgpu_kernel void @test_copy_v4i8_volatile_store(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  store volatile <4 x i8> %val, <4 x i8> addrspace(1)* %out, align 4
  ret void
}
