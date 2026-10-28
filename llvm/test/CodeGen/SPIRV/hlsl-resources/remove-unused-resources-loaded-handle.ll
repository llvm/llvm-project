; RUN: opt -S -passes=spirv-remove-unused-resources %s -o - | FileCheck %s

; Make sure the handle/global survive the removal pass since they're actually used
; Original HLSL:

; struct Data { float value; };
; ConstantBuffer<Data> data;
;
; __attribute__((noinline)) float readData() {
;   return data.value;
; }
;
; float main() : SV_Target {
;   return readData();
; }

; CHECK: @_ZL4data.0 = internal {{.*}}global target("spirv.VulkanBuffer", %Data, 2, 0)
; CHECK-LABEL: define internal {{.*}}@_Z8readDatav()
; CHECK: load target("spirv.VulkanBuffer", %Data, 2, 0), ptr @_ZL4data.0
; CHECK-LABEL: define void @main()
; CHECK: %[[HANDLE:[0-9]+]] = {{.*}}call target("spirv.VulkanBuffer", %Data, 2, 0) @llvm.spv.resource.handlefromimplicitbinding
; CHECK-NEXT: store target("spirv.VulkanBuffer", %Data, 2, 0) %[[HANDLE]], ptr @_ZL4data.0

target datalayout = "e-ve-i64:64-n8:16:32:64-G10"
target triple = "spirv1.6-unknown-vulkan1.3-pixel"

%Data = type <{ float }>

@_ZL4data.0 = internal unnamed_addr global target("spirv.VulkanBuffer", %Data, 2, 0) poison, align 8
@.str = private unnamed_addr constant [5 x i8] c"data\00", align 1
@SV_Target0 = external hidden thread_local local_unnamed_addr addrspace(8) global float, !spirv.Decorations !6

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #0

; Function Attrs: convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none, target_mem: none)
define internal spir_func noundef nofpclass(nan inf) float @_Z8readDatav() local_unnamed_addr #1 {
entry:
  %0 = tail call token @llvm.experimental.convergence.entry()
  %1 = load target("spirv.VulkanBuffer", %Data, 2, 0), ptr @_ZL4data.0, align 8, !tbaa !19
  %2 = call noundef align 1 dereferenceable(4) ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Datas_2_0t(target("spirv.VulkanBuffer", %Data, 2, 0) %1) [ "convergencectrl"(token %0) ]
  %3 = load float, ptr addrspace(12) %2, align 4, !tbaa !22
  ret float %3
}

; Function Attrs: convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: read, inaccessiblemem: none, target_mem: none)
define void @main() local_unnamed_addr #2 {
entry:
  %0 = tail call token @llvm.experimental.convergence.entry()
  %1 = tail call target("spirv.VulkanBuffer", %Data, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_s_Datas_2_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  store target("spirv.VulkanBuffer", %Data, 2, 0) %1, ptr @_ZL4data.0, align 8, !tbaa !19
  %call.i = call reassoc nnan ninf nsz arcp afn spir_func noundef nofpclass(nan inf) float @_Z8readDatav() #4 [ "convergencectrl"(token %0) ]
  store float %call.i, ptr addrspace(8) @SV_Target0, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare target("spirv.VulkanBuffer", %Data, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_s_Datas_2_0t(i32, i32, i32, i32, ptr) #3

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Datas_2_0t(target("spirv.VulkanBuffer", %Data, 2, 0)) #0

attributes #0 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none, target_mem: none) "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: read, inaccessiblemem: none, target_mem: none) "hlsl.shader"="pixel" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent }

!llvm.errno.tbaa = !{!11}

!6 = !{!7}
!7 = !{i32 30, i32 0}
!11 = !{!12, !14, i64 0}
!12 = !{!"__libc_errno", !14, i64 0}
!14 = !{!"int", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C++ TBAA"}
!19 = !{!20, !16, i64 0}
!20 = !{!"_ZTSN4hlsl14ConstantBufferI4DataEE", !16, i64 0}
!22 = !{!23, !23, i64 0}
!23 = !{!"float", !16, i64 0}
