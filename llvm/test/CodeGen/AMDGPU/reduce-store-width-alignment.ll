; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}store_v2i32_as_v4i16_align_4:
; GCN: s_load_dwordx2
; GCN: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset1:1{{$}}
define amdgpu_kernel void @store_v2i32_as_v4i16_align_4(<4 x i16> addrspace(3)* align 4 %out, <2 x i32> %x) #0 {
  %x.bc = bitcast <2 x i32> %x to <4 x i16>
  store <4 x i16> %x.bc, <4 x i16> addrspace(3)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}store_v4i32_as_v8i16_align_4:
; GCN: s_load_dwordx4
; GCN-DAG: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:2 offset1:3
; GCN-DAG: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset1:1{{$}}
define amdgpu_kernel void @store_v4i32_as_v8i16_align_4(<8 x i16> addrspace(3)* align 4 %out, <4 x i32> %x) #0 {
  %x.bc = bitcast <4 x i32> %x to <8 x i16>
  store <8 x i16> %x.bc, <8 x i16> addrspace(3)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}store_v2i32_as_i64_align_4:
; GCN: s_load_dwordx2
; GCN: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset1:1{{$}}
define amdgpu_kernel void @store_v2i32_as_i64_align_4(<4 x i16> addrspace(3)* align 4 %out, <2 x i32> %x) #0 {
  %x.bc = bitcast <2 x i32> %x to <4 x i16>
  store <4 x i16> %x.bc, <4 x i16> addrspace(3)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}store_v4i32_as_v2i64_align_4:
; GCN: s_load_dwordx4
; GCN-DAG: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:2 offset1:3
; GCN-DAG: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset1:1{{$}}
define amdgpu_kernel void @store_v4i32_as_v2i64_align_4(<2 x i64> addrspace(3)* align 4 %out, <4 x i32> %x) #0 {
  %x.bc = bitcast <4 x i32> %x to <2 x i64>
  store <2 x i64> %x.bc, <2 x i64> addrspace(3)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}store_v4i16_as_v2i32_align_4:
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset1:1{{$}}
define amdgpu_kernel void @store_v4i16_as_v2i32_align_4(<2 x i32> addrspace(3)* align 4 %out, <4 x i16> %x) #0 {
  %x.bc = bitcast <4 x i16> %x to <2 x i32>
  store <2 x i32> %x.bc, <2 x i32> addrspace(3)* %out, align 4
  ret void
}

attributes #0 = { nounwind }
