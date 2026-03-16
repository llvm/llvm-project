; Test that llvm.masked.gather and llvm.masked.scatter intrinsics are correctly
; lowered to OpMaskedGatherINTEL and OpMaskedScatterINTEL SPIR-V instructions
; when the SPV_INTEL_masked_gather_scatter extension is enabled.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; TODO: spirv-val does not support vector operands in OpConvertPtrToU and OpConvertUToPtr with SPV_INTEL_masked_gather_scatter
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability MaskedGatherScatterINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_masked_gather_scatter"

define spir_kernel void @test_gather_undef() {
; CHECK-LABEL: Begin function test_gather_undef
; CHECK: OpMaskedGatherINTEL
entry:
  %data = call <4 x i32> @llvm.masked.gather.v4i32.v4p1(<4 x ptr addrspace(1)> poison, i32 4, <4 x i1> poison, <4 x i32> poison)
  ret void
}

define spir_kernel void @test_scatter_undef() {
; CHECK-LABEL: Begin function test_scatter_undef
; CHECK: OpMaskedScatterINTEL
entry:
  call void @llvm.masked.scatter.v4i32.v4p1(<4 x i32> poison, <4 x ptr addrspace(1)> poison, i32 4, <4 x i1> poison)
  ret void
}

define spir_kernel void @test_gather_v4i32(<4 x i64> %addrs, <4 x i1> %mask, <4 x i32> %passthru) {
; CHECK-LABEL: Begin function test_gather_v4i32
; CHECK: OpMaskedGatherINTEL
entry:
  %ptrs = inttoptr <4 x i64> %addrs to <4 x ptr addrspace(1)>
  %data = call <4 x i32> @llvm.masked.gather.v4i32.v4p1(<4 x ptr addrspace(1)> %ptrs, i32 4, <4 x i1> %mask, <4 x i32> %passthru)
  ret void
}

define spir_kernel void @test_scatter_v4i32(<4 x i32> %data, <4 x i64> %addrs, <4 x i1> %mask) {
; CHECK-LABEL: Begin function test_scatter_v4i32
; CHECK: OpMaskedScatterINTEL
entry:
  %ptrs = inttoptr <4 x i64> %addrs to <4 x ptr addrspace(1)>
  call void @llvm.masked.scatter.v4i32.v4p1(<4 x i32> %data, <4 x ptr addrspace(1)> %ptrs, i32 4, <4 x i1> %mask)
  ret void
}

define spir_kernel void @test_gather_v2i64(<2 x i64> %addrs, <2 x i1> %mask, <2 x i64> %passthru) {
; CHECK-LABEL: Begin function test_gather_v2i64
; CHECK: OpMaskedGatherINTEL
entry:
  %ptrs = inttoptr <2 x i64> %addrs to <2 x ptr addrspace(1)>
  %data = call <2 x i64> @llvm.masked.gather.v2i64.v2p1(<2 x ptr addrspace(1)> %ptrs, i32 8, <2 x i1> %mask, <2 x i64> %passthru)
  ret void
}

define spir_kernel void @test_scatter_v2i64(<2 x i64> %data, <2 x i64> %addrs, <2 x i1> %mask) {
; CHECK-LABEL: Begin function test_scatter_v2i64
; CHECK: OpMaskedScatterINTEL
entry:
  %ptrs = inttoptr <2 x i64> %addrs to <2 x ptr addrspace(1)>
  call void @llvm.masked.scatter.v2i64.v2p1(<2 x i64> %data, <2 x ptr addrspace(1)> %ptrs, i32 8, <2 x i1> %mask)
  ret void
}

define spir_kernel void @test_gather_v8i32(<8 x i64> %addrs, <8 x i1> %mask, <8 x i32> %passthru) {
; CHECK-LABEL: Begin function test_gather_v8i32
; CHECK: OpMaskedGatherINTEL
entry:
  %ptrs = inttoptr <8 x i64> %addrs to <8 x ptr addrspace(1)>
  %data = call <8 x i32> @llvm.masked.gather.v8i32.v8p1(<8 x ptr addrspace(1)> %ptrs, i32 4, <8 x i1> %mask, <8 x i32> %passthru)
  ret void
}

define spir_kernel void @test_scatter_v8i32(<8 x i32> %data, <8 x i64> %addrs, <8 x i1> %mask) {
; CHECK-LABEL: Begin function test_scatter_v8i32
; CHECK: OpMaskedScatterINTEL
entry:
  %ptrs = inttoptr <8 x i64> %addrs to <8 x ptr addrspace(1)>
  call void @llvm.masked.scatter.v8i32.v8p1(<8 x i32> %data, <8 x ptr addrspace(1)> %ptrs, i32 4, <8 x i1> %mask)
  ret void
}

define spir_kernel void @test_gather_v4f32(<4 x i64> %addrs, <4 x i1> %mask, <4 x float> %passthru) {
; CHECK-LABEL: Begin function test_gather_v4f32
; CHECK: OpMaskedGatherINTEL
entry:
  %ptrs = inttoptr <4 x i64> %addrs to <4 x ptr addrspace(1)>
  %data = call <4 x float> @llvm.masked.gather.v4f32.v4p1(<4 x ptr addrspace(1)> %ptrs, i32 4, <4 x i1> %mask, <4 x float> %passthru)
  ret void
}

define spir_kernel void @test_scatter_v4f32(<4 x float> %data, <4 x i64> %addrs, <4 x i1> %mask) {
; CHECK-LABEL: Begin function test_scatter_v4f32
; CHECK: OpMaskedScatterINTEL
entry:
  %ptrs = inttoptr <4 x i64> %addrs to <4 x ptr addrspace(1)>
  call void @llvm.masked.scatter.v4f32.v4p1(<4 x float> %data, <4 x ptr addrspace(1)> %ptrs, i32 4, <4 x i1> %mask)
  ret void
}

declare <4 x i32> @llvm.masked.gather.v4i32.v4p1(<4 x ptr addrspace(1)>, i32, <4 x i1>, <4 x i32>)
declare void @llvm.masked.scatter.v4i32.v4p1(<4 x i32>, <4 x ptr addrspace(1)>, i32, <4 x i1>)
declare <2 x i64> @llvm.masked.gather.v2i64.v2p1(<2 x ptr addrspace(1)>, i32, <2 x i1>, <2 x i64>)
declare void @llvm.masked.scatter.v2i64.v2p1(<2 x i64>, <2 x ptr addrspace(1)>, i32, <2 x i1>)
declare <8 x i32> @llvm.masked.gather.v8i32.v8p1(<8 x ptr addrspace(1)>, i32, <8 x i1>, <8 x i32>)
declare void @llvm.masked.scatter.v8i32.v8p1(<8 x i32>, <8 x ptr addrspace(1)>, i32, <8 x i1>)
declare <4 x float> @llvm.masked.gather.v4f32.v4p1(<4 x ptr addrspace(1)>, i32, <4 x i1>, <4 x float>)
declare void @llvm.masked.scatter.v4f32.v4p1(<4 x float>, <4 x ptr addrspace(1)>, i32, <4 x i1>)
