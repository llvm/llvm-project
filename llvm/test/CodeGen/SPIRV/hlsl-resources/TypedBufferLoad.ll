; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; When accessing read-only `Buffer` types, SPIR-V should use `OpImageFetch` instead of `OpImageRead`.
; https://github.com/llvm/llvm-project/issues/162891

; CHECK-DAG: OpCapability SampledBuffer
; CHECK-DAG: OpCapability ImageBuffer
; CHECK-DAG: [[TypeInt:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[TypeImageBuffer:%[0-9]+]] = OpTypeImage [[TypeInt]] Buffer 2 0 0 1 Unknown
; CHECK-DAG: [[TypePtrImageBuffer:%[0-9]+]] = OpTypePointer UniformConstant [[TypeImageBuffer]]
; CHECK-DAG: [[TypeVector:%[0-9]+]] = OpTypeVector [[TypeInt]] 4
; CHECK-DAG: [[Index:%[0-9]+]] = OpConstant [[TypeInt]] 98
; CHECK-DAG: [[Variable:%[0-9]+]] = OpVariable [[TypePtrImageBuffer]] UniformConstant
@.str = private unnamed_addr constant [7 x i8] c"rwbuff\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"buff\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  %0 = tail call target("spirv.Image", i32, 5, 2, 0, 0, 2, 33) @llvm.spv.resource.handlefromimplicitbinding.tspirv.Image_i32_5_2_0_0_2_33t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %1 = tail call target("spirv.Image", i32, 5, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.Image_i32_5_2_0_0_1_0t(i32 1, i32 0, i32 1, i32 0, ptr nonnull @.str.2)
  %2 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_1_0t(target("spirv.Image", i32, 5, 2, 0, 0, 1, 0) %1, i32 98)
; CHECK: [[Load:%[0-9]+]] = OpLoad [[TypeImageBuffer]] [[Variable]]
; CHECK: [[ImageFetch:%[0-9]+]] = OpImageFetch [[TypeVector]] [[Load]] [[Index]]
; CHECK: {{.*}} = OpCompositeExtract [[TypeInt]] [[ImageFetch]] 0
  %3 = load i32, ptr addrspace(11) %2, align 4
  %4 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_2_33t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 33) %0, i32 99)
  store i32 %3, ptr addrspace(11) %4, align 4
  %5 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_2_33t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 33) %0, i32 96)
; CHECK: {{%[0-9]+}} = OpLoad {{.*}}
; CHECK: {{%[0-9]+}} = OpImageRead {{.*}}
  %6 = load i32, ptr addrspace(11) %5, align 4
  %7 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_2_33t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 33) %0, i32 97)
  store i32 %6, ptr addrspace(11) %7, align 4
  ret void
}
