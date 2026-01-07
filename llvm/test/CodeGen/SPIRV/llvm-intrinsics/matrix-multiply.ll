; RUN: llc -O0 -mtriple=spirv1.5-unknown-vulkan1.2 %s -o - | FileCheck %s --check-prefixes=CHECK,VK1_1
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.5-unknown-vulkan1.2 %s -o - -filetype=obj | spirv-val --target-env vulkan1.2 %}

; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - | FileCheck %s --check-prefixes=CHECK,VK1_3
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

@private_v4f32 = internal addrspace(10) global [4 x float] poison
@private_v4i32 = internal addrspace(10) global [4 x i32] poison
@private_v6f32 = internal addrspace(10) global [6 x float] poison
@private_v2f32 = internal addrspace(10) global [2 x float] poison
@private_v1f32 = internal addrspace(10) global [1 x float] poison


; CHECK-DAG: %[[Float_ID:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[V2F32_ID:[0-9]+]] = OpTypeVector %[[Float_ID]] 2
; CHECK-DAG: %[[V3F32_ID:[0-9]+]] = OpTypeVector %[[Float_ID]] 3
; CHECK-DAG: %[[V4F32_ID:[0-9]+]] = OpTypeVector %[[Float_ID]] 4
; CHECK-DAG: %[[Int_ID:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[V2I32_ID:[0-9]+]] = OpTypeVector %[[Int_ID]] 2
; CHECK-DAG: %[[V4I32_ID:[0-9]+]] = OpTypeVector %[[Int_ID]] 4

; Test Matrix Multiply 2x2 * 2x2 float
; CHECK-LABEL: ; -- Begin function test_matrix_multiply_f32_2x2_2x2
; CHECK:       %[[A:[0-9]+]] = OpCompositeInsert %[[V4F32_ID]] {{.*}} {{.*}} 3
; CHECK:       %[[B:[0-9]+]] = OpCompositeInsert %[[V4F32_ID]] {{.*}} {{.*}} 3
; CHECK-DAG:   %[[B_Col0:[0-9]+]] = OpVectorShuffle %[[V2F32_ID]] %[[B]] %[[#]] 0 1
; CHECK-DAG:   %[[B_Col1:[0-9]+]] = OpVectorShuffle %[[V2F32_ID]] %[[B]] %[[#]] 2 3
; CHECK-DAG:   %[[A_Row0:[0-9]+]] = OpVectorShuffle %[[V2F32_ID]] %[[A]] %[[A]] 0 2
; CHECK-DAG:   %[[A_Row1:[0-9]+]] = OpVectorShuffle %[[V2F32_ID]] %[[A]] %[[A]] 1 3
; CHECK-DAG:   %[[C00:[0-9]+]] = OpDot %[[Float_ID]] %[[A_Row0]] %[[B_Col0]]
; CHECK-DAG:   %[[C10:[0-9]+]] = OpDot %[[Float_ID]] %[[A_Row1]] %[[B_Col0]]
; CHECK-DAG:   %[[C01:[0-9]+]] = OpDot %[[Float_ID]] %[[A_Row0]] %[[B_Col1]]
; CHECK-DAG:   %[[C11:[0-9]+]] = OpDot %[[Float_ID]] %[[A_Row1]] %[[B_Col1]]
; CHECK:       OpCompositeConstruct %[[V4F32_ID]] %[[C00]] %[[C10]] %[[C01]] %[[C11]]
define internal void @test_matrix_multiply_f32_2x2_2x2() {
  %1 = load <4 x float>, ptr addrspace(10) @private_v4f32
  %2 = load <4 x float>, ptr addrspace(10) @private_v4f32
  %3 = call <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float> %1, <4 x float> %2, i32 2, i32 2, i32 2)
  store <4 x float> %3, ptr addrspace(10) @private_v4f32
  ret void
}

; Test Matrix Multiply 2x2 * 2x2 int
; CHECK-LABEL: ; -- Begin function test_matrix_multiply_i32_2x2_2x2
; CHECK:       %[[A:[0-9]+]] = OpCompositeInsert %[[V4I32_ID]] {{.*}} {{.*}} 3
; CHECK:       %[[B:[0-9]+]] = OpCompositeInsert %[[V4I32_ID]] {{.*}} {{.*}} 3
; CHECK-DAG:   %[[B_Col0:[0-9]+]] = OpVectorShuffle %[[V2I32_ID]] %[[B]] %[[#]] 0 1
; CHECK-DAG:   %[[B_Col1:[0-9]+]] = OpVectorShuffle %[[V2I32_ID]] %[[B]] %[[#]] 2 3
; CHECK-DAG:   %[[A_Row0:[0-9]+]] = OpVectorShuffle %[[V2I32_ID]] %[[A]] %[[A]] 0 2
; CHECK-DAG:   %[[A_Row1:[0-9]+]] = OpVectorShuffle %[[V2I32_ID]] %[[A]] %[[A]] 1 3
;
; -- C00 = dot(A_Row0, B_Col0)
; VK1_1-DAG:   %[[Mul00:[0-9]+]] = OpIMul %[[V2I32_ID]] %[[A_Row0]] %[[B_Col0]]
; VK1_1-DAG:   %[[E00_0:[0-9]+]] = OpCompositeExtract %[[Int_ID]] %[[Mul00]] 0
; VK1_1-DAG:   %[[E00_1:[0-9]+]] = OpCompositeExtract %[[Int_ID]] %[[Mul00]] 1
; VK1_1-DAG:   %[[C00:[0-9]+]] = OpIAdd %[[Int_ID]] %[[E00_0]] %[[E00_1]]
; VK1_3-DAG:   %[[C00:[0-9]+]] = OpUDot %[[Int_ID]] %[[A_Row0]] %[[B_Col0]]
;
; -- C10 = dot(A_Row1, B_Col0)
; VK1_1-DAG:   %[[Mul10:[0-9]+]] = OpIMul %[[V2I32_ID]] %[[A_Row1]] %[[B_Col0]]
; VK1_1-DAG:   %[[E10_0:[0-9]+]] = OpCompositeExtract %[[Int_ID]] %[[Mul10]] 0
; VK1_1-DAG:   %[[E10_1:[0-9]+]] = OpCompositeExtract %[[Int_ID]] %[[Mul10]] 1
; VK1_1-DAG:   %[[C10:[0-9]+]] = OpIAdd %[[Int_ID]] %[[E10_0]] %[[E10_1]]
; VK1_3-DAG:   %[[C10:[0-9]+]] = OpUDot %[[Int_ID]] %[[A_Row1]] %[[B_Col0]]
;
; -- C11 = dot(A_Row1, B_Col1)
; VK1_1-DAG:   %[[Mul11:[0-9]+]] = OpIMul %[[V2I32_ID]] %[[A_Row1]] %[[B_Col1]]
; VK1_1-DAG:   %[[E11_0:[0-9]+]] = OpCompositeExtract %[[Int_ID]] %[[Mul11]] 0
; VK1_1-DAG:   %[[E11_1:[0-9]+]] = OpCompositeExtract %[[Int_ID]] %[[Mul11]] 1
; VK1_1-DAG:   %[[C11:[0-9]+]] = OpIAdd %[[Int_ID]] %[[E11_0]] %[[E11_1]]
; VK1_3-DAG:   %[[C11:[0-9]+]] = OpUDot %[[Int_ID]] %[[A_Row1]] %[[B_Col1]]
;
; -- C01 = dot(A_Row0, B_Col1)
; VK1_1-DAG:   %[[Mul01:[0-9]+]] = OpIMul %[[V2I32_ID]] %[[A_Row0]] %[[B_Col1]]
; VK1_1-DAG:   %[[E01_0:[0-9]+]] = OpCompositeExtract %[[Int_ID]] %[[Mul01]] 0
; VK1_1-DAG:   %[[E01_1:[0-9]+]] = OpCompositeExtract %[[Int_ID]] %[[Mul01]] 1
; VK1_1-DAG:   %[[C01:[0-9]+]] = OpIAdd %[[Int_ID]] %[[E01_0]] %[[E01_1]]
; VK1_3-DAG:   %[[C01:[0-9]+]] = OpUDot %[[Int_ID]] %[[A_Row0]] %[[B_Col1]]
;
; CHECK:       OpCompositeConstruct %[[V4I32_ID]] %[[C00]] %[[C10]] %[[C01]] %[[C11]]
define internal void @test_matrix_multiply_i32_2x2_2x2() {
  %1 = load <4 x i32>, ptr addrspace(10) @private_v4i32
  %2 = load <4 x i32>, ptr addrspace(10) @private_v4i32
  %3 = call <4 x i32> @llvm.matrix.multiply.v4i32.v4i32.v4i32(<4 x i32> %1, <4 x i32> %2, i32 2, i32 2, i32 2)
  store <4 x i32> %3, ptr addrspace(10) @private_v4i32
  ret void
}

; Test Matrix Multiply 2x3 * 3x2 float (Result 2x2 float)
; CHECK-LABEL: ; -- Begin function test_matrix_multiply_f32_2x3_3x2
; CHECK:       %[[Col0B:[0-9]+]] = OpCompositeConstruct %[[V3F32_ID]] {{.*}} {{.*}} {{.*}}
; CHECK:       %[[Col1B:[0-9]+]] = OpCompositeConstruct %[[V3F32_ID]] {{.*}} {{.*}} {{.*}}
; CHECK:       %[[Row0A:[0-9]+]] = OpCompositeConstruct %[[V3F32_ID]] {{.*}} {{.*}} {{.*}}
; CHECK:       %[[Row1A:[0-9]+]] = OpCompositeConstruct %[[V3F32_ID]] {{.*}} {{.*}} {{.*}}
;
; CHECK-DAG:   %[[C00:[0-9]+]] = OpDot %[[Float_ID]] %[[Row0A]] %[[Col0B]]
; CHECK-DAG:   %[[C10:[0-9]+]] = OpDot %[[Float_ID]] %[[Row1A]] %[[Col0B]]
; CHECK-DAG:   %[[C01:[0-9]+]] = OpDot %[[Float_ID]] %[[Row0A]] %[[Col1B]]
; CHECK-DAG:   %[[C11:[0-9]+]] = OpDot %[[Float_ID]] %[[Row1A]] %[[Col1B]]
; CHECK:       OpCompositeConstruct %[[V4F32_ID]] %[[C00]] %[[C10]] %[[C01]] %[[C11]]
define internal void @test_matrix_multiply_f32_2x3_3x2() {
  %1 = load <6 x float>, ptr addrspace(10) @private_v6f32
  %2 = load <6 x float>, ptr addrspace(10) @private_v6f32
  %3 = call <4 x float> @llvm.matrix.multiply.v4f32.v6f32.v6f32(<6 x float> %1, <6 x float> %2, i32 2, i32 3, i32 2)
  store <4 x float> %3, ptr addrspace(10) @private_v4f32
  ret void
}

; Test Matrix Multiply 2x2 * 2x1 float (Result 2x1 vector)
; CHECK-LABEL: ; -- Begin function test_matrix_multiply_f32_2x2_2x1_vec
; CHECK:       %[[A:[0-9]+]] = OpCompositeInsert %[[V4F32_ID]] {{.*}} {{.*}} 3
; CHECK:       %[[B:[0-9]+]] = OpCompositeInsert %[[V2F32_ID]] {{.*}} {{.*}} 1
; CHECK-DAG:   %[[A_Row0:[0-9]+]] = OpVectorShuffle %[[V2F32_ID]] %[[A]] %[[A]] 0 2
; CHECK-DAG:   %[[A_Row1:[0-9]+]] = OpVectorShuffle %[[V2F32_ID]] %[[A]] %[[A]] 1 3
; CHECK-DAG:   %[[C00:[0-9]+]] = OpDot %[[Float_ID]] %[[A_Row0]] %[[B]]
; CHECK-DAG:   %[[C10:[0-9]+]] = OpDot %[[Float_ID]] %[[A_Row1]] %[[B]]
; CHECK:       OpCompositeConstruct %[[V2F32_ID]] %[[C00]] %[[C10]]
define internal void @test_matrix_multiply_f32_2x2_2x1_vec() {
  %1 = load <4 x float>, ptr addrspace(10) @private_v4f32
  %2 = load <2 x float>, ptr addrspace(10) @private_v2f32
  %3 = call <2 x float> @llvm.matrix.multiply.v2f32.v4f32.v2f32(<4 x float> %1, <2 x float> %2, i32 2, i32 2, i32 1)
  store <2 x float> %3, ptr addrspace(10) @private_v2f32
  ret void
}

; Test Matrix Multiply 1x2 * 2x2 float (Result 1x2 vector)
; CHECK-LABEL: ; -- Begin function test_matrix_multiply_f32_1x2_2x2_vec
; CHECK:       %[[A:[0-9]+]] = OpCompositeInsert %[[V2F32_ID]] {{.*}} {{.*}} 1
; CHECK:       %[[B:[0-9]+]] = OpCompositeInsert %[[V4F32_ID]] {{.*}} {{.*}} 3
; CHECK-DAG:   %[[B_Col0:[0-9]+]] = OpVectorShuffle %[[V2F32_ID]] %[[B]] %[[#]] 0 1
; CHECK-DAG:   %[[B_Col1:[0-9]+]] = OpVectorShuffle %[[V2F32_ID]] %[[B]] %[[#]] 2 3
; CHECK-DAG:   %[[C00:[0-9]+]] = OpDot %[[Float_ID]] %[[A]] %[[B_Col0]]
; CHECK-DAG:   %[[C01:[0-9]+]] = OpDot %[[Float_ID]] %[[A]] %[[B_Col1]]
; CHECK:       OpCompositeConstruct %[[V2F32_ID]] %[[C00]] %[[C01]]
define internal void @test_matrix_multiply_f32_1x2_2x2_vec() {
  %1 = load <2 x float>, ptr addrspace(10) @private_v2f32
  %2 = load <4 x float>, ptr addrspace(10) @private_v4f32
  %3 = call <2 x float> @llvm.matrix.multiply.v2f32.v2f32.v4f32(<2 x float> %1, <4 x float> %2, i32 1, i32 2, i32 2)
  store <2 x float> %3, ptr addrspace(10) @private_v2f32
  ret void
}

; Test Matrix Multiply 1x2 * 2x1 float (Result 1x1 scalar - OpDot)
; TODO(171175): The SPIR-V backend does not legalize single element vectors.
; CHECK-DISABLE: ; -- Begin function test_matrix_multiply_f32_1x2_2x1_scalar
; define internal void @test_matrix_multiply_f32_1x2_2x1_scalar() {
;   %1 = load <2 x float>, ptr addrspace(10) @private_v2f32
;   %2 = load <2 x float>, ptr addrspace(10) @private_v2f32
;   %3 = call <1 x float> @llvm.matrix.multiply.v1f32.v2f32.v2f32(<2 x float> %1, <2 x float> %2, i32 1, i32 2, i32 1)
;   store <1 x float> %3, ptr addrspace(10) @private_v1f32
;   ret void
; }

define void @main() #0 {
  ret void
}

declare <4 x float> @llvm.matrix.multiply.v4f32.v4f32.v4f32(<4 x float>, <4 x float>, i32, i32, i32)
declare <4 x i32> @llvm.matrix.multiply.v4i32.v4i32.v4i32(<4 x i32>, <4 x i32>, i32, i32, i32)
declare <4 x float> @llvm.matrix.multiply.v4f32.v6f32.v6f32(<6 x float>, <6 x float>, i32, i32, i32)
declare <2 x float> @llvm.matrix.multiply.v2f32.v4f32.v2f32(<4 x float>, <2 x float>, i32, i32, i32)
declare <2 x float> @llvm.matrix.multiply.v2f32.v2f32.v4f32(<2 x float>, <4 x float>, i32, i32, i32)
; declare <1 x float> @llvm.matrix.multiply.v1f32.v2f32.v2f32(<2 x float>, <2 x float>, i32, i32, i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
