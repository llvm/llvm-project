; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -stop-after=spirv-legalize-pointer-cast -o - | FileCheck %s --check-prefix=IRCHECK
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: [[FLOAT:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[VEC2FLOAT:%[0-9]+]] = OpTypeVector [[FLOAT]] 2
; CHECK: OpLoad [[VEC2FLOAT]]
; CHECK: OpCompositeExtract [[FLOAT]]
; CHECK: OpCompositeConstruct [[VEC2FLOAT]]

; // M1[0][0]
; IRCHECK: [[M1ROWZero:%[0-9]+]] = call ptr addrspace(10) (i1, ptr addrspace(10), ...) @llvm.spv.gep.p10.p10(i1 false, ptr addrspace(10) @M1, i32 0, i32 0)
; IRCHECK: [[M1ROWZeroVec:%[0-9]+]] = load <2 x float>, ptr addrspace(10) [[M1ROWZero]], align 4
; IRCHECK: [[M1Elem_00:%[0-9]+]] = call float @llvm.spv.extractelt.f32.v2f32.i32(<2 x float> [[M1ROWZeroVec]], i32 0)

;// M1[0][1]
; IRCHECK: [[M1ROWZero_2:%[0-9]+]] = call ptr addrspace(10) (i1, ptr addrspace(10), ...) @llvm.spv.gep.p10.p10(i1 false, ptr addrspace(10) @M1, i32 0, i32 0)
; IRCHECK: [[M1ROWZeroVec_2:%[0-9]+]] = load <2 x float>, ptr addrspace(10) [[M1ROWZero_2]], align 4
; IRCHECK: [[M1Elem_01:%[0-9]+]] = call float @llvm.spv.extractelt.f32.v2f32.i32(<2 x float> [[M1ROWZeroVec_2]], i32 1)

; // M1[1][0]
; IRCHECK: [[M1ROWOne:%[0-9]+]] = call ptr addrspace(10) (i1, ptr addrspace(10), ...) @llvm.spv.gep.p10.p10(i1 false, ptr addrspace(10) @M1, i32 0, i32 1)
; IRCHECK: [[M1ROWOneVec:%[0-9]+]] = load <2 x float>, ptr addrspace(10) [[M1ROWOne]], align 4
; IRCHECK: [[M1Elem_10:%[0-9]+]] = call float @llvm.spv.extractelt.f32.v2f32.i32(<2 x float> [[M1ROWOneVec]], i32 0)

; // M1[1][1]
; IRCHECK: [[M1ROWOne_2:%[0-9]+]] = call ptr addrspace(10) (i1, ptr addrspace(10), ...) @llvm.spv.gep.p10.p10(i1 false, ptr addrspace(10) @M1, i32 0, i32 1)
; IRCHECK: [[M1ROWOneVec_2:%[0-9]+]] = load <2 x float>, ptr addrspace(10) [[M1ROWOne_2]], align 4
; IRCHECK: [[M1Elem_11:%[0-9]+]] = call float @llvm.spv.extractelt.f32.v2f32.i32(<2 x float> [[M1ROWOneVec_2]], i32 1)

; // M1[2][0]
; IRCHECK: [[M1ROWTwo:%[0-9]+]] = call ptr addrspace(10) (i1, ptr addrspace(10), ...) @llvm.spv.gep.p10.p10(i1 false, ptr addrspace(10) @M1, i32 0, i32 2)
; IRCHECK: [[M1ROWTwoVec:%[0-9]+]] = load <2 x float>, ptr addrspace(10) [[M1ROWTwo]], align 4
; IRCHECK: [[M1Elem_20:%[0-9]+]] = call float @llvm.spv.extractelt.f32.v2f32.i32(<2 x float> [[M1ROWTwoVec]], i32 0)

@M1 = internal addrspace(10) global [4 x <2 x float>] zeroinitializer, align 4
@OUT = internal addrspace(10) global <2 x float> zeroinitializer, align 4

define spir_func void @main() #1 {
entry:
  %0 = load <5 x float>, ptr addrspace(10) @M1, align 4
  %1 = shufflevector <5 x float> %0, <5 x float> poison, <2 x i32> <i32 0, i32 4>
  store <2 x float> %1, ptr addrspace(10) @OUT, align 4
  ret void
}

attributes #1 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
