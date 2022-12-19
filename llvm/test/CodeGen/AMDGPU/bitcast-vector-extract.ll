; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; The bitcast should be pushed through the bitcasts so the vectors can
; be broken down and the shared components can be CSEd

; GCN-LABEL: {{^}}store_bitcast_constant_v8i32_to_v8f32:
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN-NOT: v_mov_b32
; GCN: buffer_store_dwordx4
; GCN-NOT: v_mov_b32
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @store_bitcast_constant_v8i32_to_v8f32(ptr addrspace(1) %out, <8 x i32> %vec) {
  %vec0.bc = bitcast <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 8> to <8 x float>
  store volatile <8 x float> %vec0.bc, ptr addrspace(1) %out

  %vec1.bc = bitcast <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 9> to <8 x float>
  store volatile <8 x float> %vec1.bc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_bitcast_constant_v4i64_to_v8f32:
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN-NOT: v_mov_b32
; GCN: buffer_store_dwordx4
; GCN-NOT: v_mov_b32
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @store_bitcast_constant_v4i64_to_v8f32(ptr addrspace(1) %out, <4 x i64> %vec) {
  %vec0.bc = bitcast <4 x i64> <i64 7, i64 7, i64 7, i64 8> to <8 x float>
  store volatile <8 x float> %vec0.bc, ptr addrspace(1) %out

  %vec1.bc = bitcast <4 x i64> <i64 7, i64 7, i64 7, i64 9> to <8 x float>
  store volatile <8 x float> %vec1.bc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_bitcast_constant_v4i64_to_v4f64:
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN-NOT: v_mov_b32
; GCN: buffer_store_dwordx4
; GCN-NOT: v_mov_b32
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @store_bitcast_constant_v4i64_to_v4f64(ptr addrspace(1) %out, <4 x i64> %vec) {
  %vec0.bc = bitcast <4 x i64> <i64 7, i64 7, i64 7, i64 8> to <4 x double>
  store volatile <4 x double> %vec0.bc, ptr addrspace(1) %out

  %vec1.bc = bitcast <4 x i64> <i64 7, i64 7, i64 7, i64 9> to <4 x double>
  store volatile <4 x double> %vec1.bc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_bitcast_constant_v8i32_to_v16i16:
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN-NOT: v_mov_b32
; GCN: buffer_store_dwordx4
; GCN-NOT: v_mov_b32
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @store_bitcast_constant_v8i32_to_v16i16(ptr addrspace(1) %out, <16 x i16> %vec) {
  %vec0.bc = bitcast <16 x i16> <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 8> to <8 x float>
  store volatile <8 x float> %vec0.bc, ptr addrspace(1) %out

  %vec1.bc = bitcast <16 x i16> <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 9> to <8 x float>
  store volatile <8 x float> %vec1.bc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_value_lowered_to_undef_bitcast_source:
; GCN-NOT: store_dword
define amdgpu_kernel void @store_value_lowered_to_undef_bitcast_source(ptr addrspace(1) %out, i64 %a, i64 %b) #0 {
  %undef = call i64 @llvm.amdgcn.icmp.i64(i64 %a, i64 %b, i32 999) #1
  %bc = bitcast i64 %undef to <2 x i32>
  store volatile <2 x i32> %bc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_value_lowered_to_undef_bitcast_source_extractelt:
; GCN-NOT: store_dword
define amdgpu_kernel void @store_value_lowered_to_undef_bitcast_source_extractelt(ptr addrspace(1) %out, i64 %a, i64 %b) #0 {
  %undef = call i64 @llvm.amdgcn.icmp.i64(i64 %a, i64 %b, i32 9999) #1
  %bc = bitcast i64 %undef to <2 x i32>
  %elt1 = extractelement <2 x i32> %bc, i32 1
  store volatile i32 %elt1, ptr addrspace(1) %out
  ret void
}

declare i64 @llvm.amdgcn.icmp.i64(i64, i64, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone convergent }
