; RUN: opt -S -passes=spirv-remove-unused-resources %s -o - | FileCheck %s --implicit-check-not=@_ZL6unused.
;
; When using entry point main, eliminate the unused buffer handle and
; it's associated counter handle
; Original HLSL:
;
; RWStructuredBuffer<float> unused;
; RWBuffer<float> used;
;
; [numthreads(1, 1, 1)]
; void main() {
;   used[0] = 1.0;
; }
;
; [numthreads(1, 1, 1)]
; void other() {
;   unused[unused.IncrementCounter()] = 2.0;
; }

; CHECK-LABEL: define void @main()
; CHECK-NOT: call target("spirv.VulkanBuffer",
; CHECK: %[[USED:[0-9]+]] = {{.*}}call target("spirv.Image", {{.*}}) @llvm.spv.resource.handlefromimplicitbinding{{.*}}(i32 2, i32 0, i32 1, i32 0, ptr nonnull @.str.2)
; CHECK: @llvm.spv.resource.getpointer{{.*}}(target("spirv.Image", {{.*}}) %[[USED]], i32 0)
; CHECK: store float 1.000000e+00
; CHECK: ret void

target datalayout = "e-ve-i64:64-n8:16:32:64-G10"
target triple = "spirv1.6-unknown-vulkan1.3-compute"

@_ZL6unused.0 = internal unnamed_addr global target("spirv.VulkanBuffer", [0 x float], 12, 1) poison, align 8
@_ZL6unused.1 = internal unnamed_addr global target("spirv.VulkanBuffer", i32, 12, 1) poison, align 8
@.str = private unnamed_addr constant [7 x i8] c"unused\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"used\00", align 1

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #0

; Function Attrs: convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, inaccessiblemem: none, target_mem: none)
define void @main() local_unnamed_addr #1 {
entry:
  %0 = tail call token @llvm.experimental.convergence.entry()
  %1 = tail call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_a0f32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %2 = call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromimplicitbinding.tspirv.VulkanBuffer_i32_12_1t.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %1, i32 1, i32 0) [ "convergencectrl"(token %0) ]
  store target("spirv.VulkanBuffer", [0 x float], 12, 1) %1, ptr @_ZL6unused.0, align 8, !tbaa !16
  store target("spirv.VulkanBuffer", i32, 12, 1) %2, ptr @_ZL6unused.1, align 8, !tbaa !18
  %3 = tail call target("spirv.Image", float, 5, 2, 0, 0, 2, 3) @llvm.spv.resource.handlefromimplicitbinding.tspirv.Image_f32_5_2_0_0_2_3t(i32 2, i32 0, i32 1, i32 0, ptr nonnull @.str.2)
  %4 = call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_3t.i32(target("spirv.Image", float, 5, 2, 0, 0, 2, 3) %3, i32 0) [ "convergencectrl"(token %0) ]
  store float 1.000000e+00, ptr addrspace(11) %4, align 4, !tbaa !20
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_a0f32_12_1t(i32, i32, i32, i32, ptr) #2

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromimplicitbinding.tspirv.VulkanBuffer_i32_12_1t.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1), i32, i32) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare target("spirv.Image", float, 5, 2, 0, 0, 2, 3) @llvm.spv.resource.handlefromimplicitbinding.tspirv.Image_f32_5_2_0_0_2_3t(i32, i32, i32, i32, ptr) #2

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_3t.i32(target("spirv.Image", float, 5, 2, 0, 0, 2, 3), i32) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare i32 @llvm.spv.resource.updatecounter.tspirv.VulkanBuffer_i32_12_1t(target("spirv.VulkanBuffer", i32, 12, 1), i8) #3

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t.i32(target("spirv.VulkanBuffer", [0 x float], 12, 1), i32) #0

attributes #0 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(write, inaccessiblemem: none, target_mem: none) "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }

!llvm.errno.tbaa = !{!8}

!8 = !{!9, !11, i64 0}
!9 = !{!"__libc_errno", !11, i64 0}
!11 = !{!"int", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C++ TBAA"}
!16 = !{!17, !13, i64 0}
!17 = !{!"_ZTSN4hlsl18RWStructuredBufferIfEE", !13, i64 0, !13, i64 8}
!18 = !{!17, !13, i64 8}
!20 = !{!21, !21, i64 0}
!21 = !{!"float", !13, i64 0}
