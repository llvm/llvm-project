; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

@private_v4f32 = internal addrspace(10) global [4 x float] poison
@private_v6f32 = internal addrspace(10) global [6 x float] poison
@private_v1f32 = internal addrspace(10) global [1 x float] poison

; CHECK-DAG: %[[Float_ID:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[V4F32_ID:[0-9]+]] = OpTypeVector %[[Float_ID]] 4

; Test Transpose 2x2 float
; CHECK-LABEL: ; -- Begin function test_transpose_f32_2x2
; CHECK: %[[Shuffle:[0-9]+]] = OpVectorShuffle %[[V4F32_ID]] {{.*}} 0 2 1 3
define internal void @test_transpose_f32_2x2() {
 %1 = load <4 x float>, ptr addrspace(10) @private_v4f32
 %2 = call <4 x float> @llvm.matrix.transpose.v4f32.i32(<4 x float> %1, i32 2, i32 2)
 store <4 x float> %2, ptr addrspace(10) @private_v4f32
 ret void
}

; Test Transpose 2x3 float (Result is 3x2 float)
; Note: We should add more code to the prelegalizer combiner to be able to remove the insert and extracts. 
;       This test should reduce to a series of access chains, loads, and stores.
; CHECK-LABEL: ; -- Begin function test_transpose_f32_2x3
define internal void @test_transpose_f32_2x3() {
; -- Load input 2x3 matrix elements
; CHECK: %[[AccessChain1:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID:[0-9]+]] %[[private_v6f32:[0-9]+]] %[[int_0:[0-9]+]]
; CHECK: %[[Load1:[0-9]+]] = OpLoad %[[Float_ID]] %[[AccessChain1]]
; CHECK: %[[AccessChain2:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID]] %[[private_v6f32]] %[[int_1:[0-9]+]]
; CHECK: %[[Load2:[0-9]+]] = OpLoad %[[Float_ID]] %[[AccessChain2]]
; CHECK: %[[AccessChain3:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID]] %[[private_v6f32]] %[[int_2:[0-9]+]]
; CHECK: %[[Load3:[0-9]+]] = OpLoad %[[Float_ID]] %[[AccessChain3]]
; CHECK: %[[AccessChain4:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID]] %[[private_v6f32]] %[[int_3:[0-9]+]]
; CHECK: %[[Load4:[0-9]+]] = OpLoad %[[Float_ID]] %[[AccessChain4]]
; CHECK: %[[AccessChain5:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID]] %[[private_v6f32]] %[[int_4:[0-9]+]]
; CHECK: %[[Load5:[0-9]+]] = OpLoad %[[Float_ID]] %[[AccessChain5]]
; CHECK: %[[AccessChain6:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID]] %[[private_v6f32]] %[[int_5:[0-9]+]]
; CHECK: %[[Load6:[0-9]+]] = OpLoad %[[Float_ID]] %[[AccessChain6]]
;
; -- Construct intermediate vectors
; CHECK: %[[Construct1:[0-9]+]] = OpCompositeConstruct %[[V4F32_ID]] %[[Load1]] %[[Load2]] %[[Load3]] %[[Load4]]
; CHECK: %[[Construct2:[0-9]+]] = OpCompositeConstruct %[[V4F32_ID]] %[[Load1]] %[[Load2]] %[[Load3]] %[[Load4]]
; CHECK: %[[Construct3:[0-9]+]] = OpCompositeConstruct %[[V4F32_ID]] %[[Load5]] %[[Load6]] {{.*}} {{.*}}
; CHECK: %[[Construct4:[0-9]+]] = OpCompositeConstruct %[[V4F32_ID]] %[[Load1]] %[[Load2]] %[[Load3]] %[[Load4]]
; CHECK: %[[Construct5:[0-9]+]] = OpCompositeConstruct %[[V4F32_ID]] %[[Load1]] %[[Load2]] %[[Load3]] %[[Load4]]
; CHECK: %[[Construct6:[0-9]+]] = OpCompositeConstruct %[[V4F32_ID]] %[[Load5]] %[[Load6]] {{.*}} {{.*}}
  %1 = load <6 x float>, ptr addrspace(10) @private_v6f32

; -- Extract elements for transposition
; CHECK: %[[Extract1:[0-9]+]] = OpCompositeExtract %[[Float_ID]] %[[Construct1]] 0
; CHECK: %[[Extract2:[0-9]+]] = OpCompositeExtract %[[Float_ID]] %[[Construct2]] 2
; CHECK: %[[Extract3:[0-9]+]] = OpCompositeExtract %[[Float_ID]] %[[Construct3]] 0
; CHECK: %[[Extract4:[0-9]+]] = OpCompositeExtract %[[Float_ID]] %[[Construct4]] 1
; CHECK: %[[Extract5:[0-9]+]] = OpCompositeExtract %[[Float_ID]] %[[Construct5]] 3
; CHECK: %[[Extract6:[0-9]+]] = OpCompositeExtract %[[Float_ID]] %[[Construct6]] 1
  %2 = call <6 x float> @llvm.matrix.transpose.v6f32.i32(<6 x float> %1, i32 2, i32 3)

; -- Store output 3x2 matrix elements
; CHECK: %[[AccessChain7:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID]] %[[private_v6f32]] %[[int_0]]
; CHECK: OpStore %[[AccessChain7]] %[[Extract1]]
; CHECK: %[[AccessChain8:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID]] %[[private_v6f32]] %[[int_1]]
; CHECK: OpStore %[[AccessChain8]] %[[Extract2]]
; CHECK: %[[AccessChain9:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID]] %[[private_v6f32]] %[[int_2]]
; CHECK: OpStore %[[AccessChain9]] %[[Extract3]]
; CHECK: %[[AccessChain10:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID]] %[[private_v6f32]] %[[int_3]]
; CHECK: OpStore %[[AccessChain10]] %[[Extract4]]
; CHECK: %[[AccessChain11:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID]] %[[private_v6f32]] %[[int_4]]
; CHECK: OpStore %[[AccessChain11]] %[[Extract5]]
; CHECK: %[[AccessChain12:[0-9]+]] = OpAccessChain %[[_ptr_Float_ID]] %[[private_v6f32]] %[[int_5]]
; CHECK: OpStore %[[AccessChain12]] %[[Extract6]]
  store <6 x float> %2, ptr addrspace(10) @private_v6f32
  ret void
}

; Test Transpose 1x4 float (Result is 4x1 float), should be a copy (vector of 4 floats)
; CHECK-LABEL: ; -- Begin function test_transpose_f32_1x4_to_4x1
; CHECK: %[[Shuffle:[0-9]+]] = OpVectorShuffle %[[V4F32_ID]] {{.*}} 0 1 2 3
define internal void @test_transpose_f32_1x4_to_4x1() {
 %1 = load <4 x float>, ptr addrspace(10) @private_v4f32
 %2 = call <4 x float> @llvm.matrix.transpose.v4f32.i32(<4 x float> %1, i32 1, i32 4)
 store <4 x float> %2, ptr addrspace(10) @private_v4f32
 ret void
}

; Test Transpose 4x1 float (Result is 1x4 float), should be a copy (vector of 4 floats)
; CHECK-LABEL: ; -- Begin function test_transpose_f32_4x1_to_1x4
; CHECK: %[[Shuffle:[0-9]+]] = OpVectorShuffle %[[V4F32_ID]] {{.*}} 0 1 2 3
define internal void @test_transpose_f32_4x1_to_1x4() {
 %1 = load <4 x float>, ptr addrspace(10) @private_v4f32
 %2 = call <4 x float> @llvm.matrix.transpose.v4f32.i32(<4 x float> %1, i32 4, i32 1)
 store <4 x float> %2, ptr addrspace(10) @private_v4f32
 ret void
}

; Test Transpose 1x1 float (Result is 1x1 float), should be a copy (scalar float)
; TODO(171175): The SPIR-V backend does not seem to be legalizing single element vectors.
; define internal void @test_transpose_f32_1x1() {
;   %1 = load <1 x float>, ptr addrspace(10) @private_v1f32
;   %2 = call <1 x float> @llvm.matrix.transpose.v1f32.i32(<1 x float> %1, i32 1, i32 1)
;   store <1 x float> %2, ptr addrspace(10) @private_v1f32
;   ret void
; }

define void @main() #0 {
  ret void
}

declare <4 x float> @llvm.matrix.transpose.v4f32.i32(<4 x float>, i32, i32)
declare <6 x float> @llvm.matrix.transpose.v6f32.i32(<6 x float>, i32, i32)
; declare <1 x float> @llvm.matrix.transpose.v1f32.i32(<1 x float>, i32, i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
