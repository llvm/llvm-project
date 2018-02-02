; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI,FUNC %s

declare i16 @llvm.bswap.i16(i16) nounwind readnone
declare i32 @llvm.bswap.i32(i32) nounwind readnone
declare <2 x i32> @llvm.bswap.v2i32(<2 x i32>) nounwind readnone
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>) nounwind readnone
declare <8 x i32> @llvm.bswap.v8i32(<8 x i32>) nounwind readnone
declare i64 @llvm.bswap.i64(i64) nounwind readnone
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>) nounwind readnone
declare <4 x i64> @llvm.bswap.v4i64(<4 x i64>) nounwind readnone

; FUNC-LABEL: @test_bswap_i32
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN-DAG: v_alignbit_b32 [[TMP0:v[0-9]+]], [[VAL]], [[VAL]], 8
; GCN-DAG: v_alignbit_b32 [[TMP1:v[0-9]+]], [[VAL]], [[VAL]], 24
; GCN-DAG: s_mov_b32 [[K:s[0-9]+]], 0xff00ff
; GCN: v_bfi_b32 [[RESULT:v[0-9]+]], [[K]], [[TMP1]], [[TMP0]]
; GCN: buffer_store_dword [[RESULT]]
; GCN: s_endpgm
define amdgpu_kernel void @test_bswap_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %bswap = call i32 @llvm.bswap.i32(i32 %val) nounwind readnone
  store i32 %bswap, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_bswap_v2i32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN: s_endpgm
define amdgpu_kernel void @test_bswap_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) nounwind {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %in, align 8
  %bswap = call <2 x i32> @llvm.bswap.v2i32(<2 x i32> %val) nounwind readnone
  store <2 x i32> %bswap, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @test_bswap_v4i32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN: s_endpgm
define amdgpu_kernel void @test_bswap_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) nounwind {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %in, align 16
  %bswap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %val) nounwind readnone
  store <4 x i32> %bswap, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: @test_bswap_v8i32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_alignbit_b32
; GCN-DAG: v_bfi_b32
; GCN: s_endpgm
define amdgpu_kernel void @test_bswap_v8i32(<8 x i32> addrspace(1)* %out, <8 x i32> addrspace(1)* %in) nounwind {
  %val = load <8 x i32>, <8 x i32> addrspace(1)* %in, align 32
  %bswap = call <8 x i32> @llvm.bswap.v8i32(<8 x i32> %val) nounwind readnone
  store <8 x i32> %bswap, <8 x i32> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: {{^}}test_bswap_i64:
; GCN-NOT: v_or_b32_e64 v{{[0-9]+}}, 0, 0
define amdgpu_kernel void @test_bswap_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) nounwind {
  %val = load i64, i64 addrspace(1)* %in, align 8
  %bswap = call i64 @llvm.bswap.i64(i64 %val) nounwind readnone
  store i64 %bswap, i64 addrspace(1)* %out, align 8
  ret void
}

define amdgpu_kernel void @test_bswap_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) nounwind {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* %in, align 16
  %bswap = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %val) nounwind readnone
  store <2 x i64> %bswap, <2 x i64> addrspace(1)* %out, align 16
  ret void
}

define amdgpu_kernel void @test_bswap_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) nounwind {
  %val = load <4 x i64>, <4 x i64> addrspace(1)* %in, align 32
  %bswap = call <4 x i64> @llvm.bswap.v4i64(<4 x i64> %val) nounwind readnone
  store <4 x i64> %bswap, <4 x i64> addrspace(1)* %out, align 32
  ret void
}

; GCN-LABEL: {{^}}missing_truncate_promote_bswap:
; VI: v_and_b32
; VI: v_alignbit_b32
; VI: v_alignbit_b32
; VI: v_bfi_b32
define float @missing_truncate_promote_bswap(i32 %arg) {
bb:
  %tmp = trunc i32 %arg to i16
  %tmp1 = call i16 @llvm.bswap.i16(i16 %tmp)
  %tmp2 = bitcast i16 %tmp1 to half
  %tmp3 = fpext half %tmp2 to float
  ret float %tmp3
}
