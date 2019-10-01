; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}global_store_v3i64:
; GCN-DAG: buffer_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:16
; GCN-DAG: buffer_store_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
define amdgpu_kernel void @global_store_v3i64(<3 x i64> addrspace(1)* %out, <3 x i64> %x) {
  store <3 x i64> %x, <3 x i64> addrspace(1)* %out, align 32
  ret void
}

; GCN-LABEL: {{^}}global_store_v3i64_unaligned:
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte

; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte

; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte

; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte

; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte

; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @global_store_v3i64_unaligned(<3 x i64> addrspace(1)* %out, <3 x i64> %x) {
  store <3 x i64> %x, <3 x i64> addrspace(1)* %out, align 1
  ret void
}

; GCN-LABEL: {{^}}local_store_v3i64:
; GCN: ds_write2_b64
; GCN: ds_write_b64
define amdgpu_kernel void @local_store_v3i64(<3 x i64> addrspace(3)* %out, <3 x i64> %x) {
  store <3 x i64> %x, <3 x i64> addrspace(3)* %out, align 32
  ret void
}

; GCN-LABEL: {{^}}local_store_v3i64_unaligned:
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8

; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8

; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8

; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8

; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8

; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
define amdgpu_kernel void @local_store_v3i64_unaligned(<3 x i64> addrspace(3)* %out, <3 x i64> %x) {
  store <3 x i64> %x, <3 x i64> addrspace(3)* %out, align 1
  ret void
}

; GCN-LABEL: {{^}}global_truncstore_v3i64_to_v3i32:
; SI-DAG: buffer_store_dwordx2
; SI-DAG: buffer_store_dword v
; VI-DAG: buffer_store_dwordx3
define amdgpu_kernel void @global_truncstore_v3i64_to_v3i32(<3 x i32> addrspace(1)* %out, <3 x i64> %x) {
  %trunc = trunc <3 x i64> %x to <3 x i32>
  store <3 x i32> %trunc, <3 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}global_truncstore_v3i64_to_v3i16:
; GCN-DAG: buffer_store_short
; GCN-DAG: buffer_store_dword v
define amdgpu_kernel void @global_truncstore_v3i64_to_v3i16(<3 x i16> addrspace(1)* %out, <3 x i64> %x) {
  %trunc = trunc <3 x i64> %x to <3 x i16>
  store <3 x i16> %trunc, <3 x i16> addrspace(1)* %out
  ret void
}


; GCN-LABEL: {{^}}global_truncstore_v3i64_to_v3i8:
; GCN-DAG: buffer_store_short
; GCN-DAG: buffer_store_byte v
define amdgpu_kernel void @global_truncstore_v3i64_to_v3i8(<3 x i8> addrspace(1)* %out, <3 x i64> %x) {
  %trunc = trunc <3 x i64> %x to <3 x i8>
  store <3 x i8> %trunc, <3 x i8> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}global_truncstore_v3i64_to_v3i1:
; GCN-DAG: buffer_store_byte v
; GCN-DAG: buffer_store_byte v
; GCN-DAG: buffer_store_byte v
define amdgpu_kernel void @global_truncstore_v3i64_to_v3i1(<3 x i1> addrspace(1)* %out, <3 x i64> %x) {
  %trunc = trunc <3 x i64> %x to <3 x i1>
  store <3 x i1> %trunc, <3 x i1> addrspace(1)* %out
  ret void
}
