; RUN: opt -S -passes=spirv-lower-resource-names,spirv-legalize-resource-binding -mtriple=spirv1.6-vulkan1.3-library < %s | FileCheck %s

@.str.used = private unnamed_addr constant [5 x i8] c"used\00", align 1
@.str = private unnamed_addr constant [23 x i8] c"ResourceDescriptorHeap\00", align 1

; Verify that an existing string does not affect the generated heap name.

; CHECK-LABEL: define void @main(
define void @main() local_unnamed_addr #0 {
entry:

; Heap resource - bound to resource array at desc 0 binding 0
; CHECK: call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.mdname{{.*}}(i32 0, i32 0, i32 0, i32 13, metadata ![[NAME:[0-9]+]])
  %heap_resource = call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromheap.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 13)

  ret void
}

; CHECK-NOT: @llvm.spv.resource.handlefromheap
; CHECK-NOT: @llvm.spv.resource.counterhandlefromheap

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

; CHECK: ![[NAME]] = !{!".ResourceDescriptorHeap.0"}
