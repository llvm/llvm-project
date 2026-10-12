; RUN: opt -S -passes=spirv-legalize-resource-binding -mtriple=spirv1.6-vulkan1.3-library < %s | FileCheck %s

@.str.used = private unnamed_addr constant [5 x i8] c"used\00", align 1
@.str = private unnamed_addr constant [23 x i8] c"ResourceDescriptorHeap\00", align 1

; Verify that a unique string is created for the heap name if the constant
; string "ResourceDescriptorHeap" already exists in the module.

; CHECK-DAG: @ResourceDescriptorHeap.1.str = private unnamed_addr constant [25 x i8] c"ResourceDescriptorHeap.1\00", align 1
; CHECK-NOT: @ResourceDescriptorHeap.str = private unnamed_addr constant [23 x i8] c"ResourceDescriptorHeap\00", align 1

; CHECK-LABEL: define void @main(
define void @main() local_unnamed_addr #0 {
entry:

; Heap resource - bound to resource array at desc 0 binding 0
; CHECK: call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 0, i32 0, i32 13, ptr @ResourceDescriptorHeap.1.str)
  %heap_resource = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromheap.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 13)

  ret void
}

; CHECK-NOT: @llvm.spv.resource.handlefromheap
; CHECK-NOT: @llvm.spv.resource.counterhandlefromheap

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }