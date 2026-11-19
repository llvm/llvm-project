;; kernel void test(global uint* dst)
;; {
;;    __spirv_ControlBarrier(1, 1, 264);  // local
;;    __spirv_ControlBarrier(1, 1, 520);  // global
;;    __spirv_ControlBarrier(1, 1, 2056); // image
;;
;;    __spirv_ControlBarrier(1, 0, 520);  // global, all_svm_devices
;;    __spirv_ControlBarrier(1, 1, 520);  // global, device
;;    __spirv_ControlBarrier(1, 2, 520);  // global, work_group
;;    __spirv_ControlBarrier(1, 3, 520);  // global, subgroup
;;    __spirv_ControlBarrier(1, 4, 520);  // global, work_item
;;}

; Test for SPV_INTEL_device_barrier (SPIR-V friendly LLVM IR)
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_barrier %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_barrier %s -o - -filetype=obj | spirv-val %}

; ModuleID = 'device_barrier_spirv.cl'
source_filename = "device_barrier_spirv.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK-SPIRV: OpCapability DeviceBarrierINTEL
; CHECK-SPIRV: OpExtension "SPV_INTEL_device_barrier"
; CHECK-SPIRV: %[[#UINT:]] = OpTypeInt 32 0
;
; Scopes:
; CHECK-SPIRV-DAG: %[[#SCOPE_CROSS_DEVICE:]] = OpConstantNull %[[#UINT]]
; CHECK-SPIRV-DAG: %[[#SCOPE_DEVICE:]] = OpConstant %[[#UINT]] 1{{$}}
; CHECK-SPIRV-DAG: %[[#SCOPE_WORK_GROUP:]] = OpConstant %[[#UINT]] 2{{$}}
; CHECK-SPIRV-DAG: %[[#SCOPE_SUBGROUP:]] = OpConstant %[[#UINT]] 3{{$}}
; CHECK-SPIRV-DAG: %[[#SCOPE_INVOCATION:]] = OpConstant %[[#UINT]] 4{{$}}
;
; Memory Semantics:
; 0x8 AcquireRelease + 0x100 WorkgroupMemory
; CHECK-SPIRV-DAG: %[[#ACQREL_LOCAL:]] = OpConstant %[[#UINT]] 264
; 0x8 AcquireRelease + 0x200 CrossWorkgroupMemory
; CHECK-SPIRV-DAG: %[[#ACQREL_GLOBAL:]] = OpConstant %[[#UINT]] 520
; 0x8 AcquireRelease + 0x800 ImageMemory
; CHECK-SPIRV-DAG: %[[#ACQREL_IMAGE:]] = OpConstant %[[#UINT]] 2056
;
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_DEVICE]] %[[#SCOPE_DEVICE]] %[[#ACQREL_LOCAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_DEVICE]] %[[#SCOPE_DEVICE]] %[[#ACQREL_GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_DEVICE]] %[[#SCOPE_DEVICE]] %[[#ACQREL_IMAGE]]
;
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_DEVICE]] %[[#SCOPE_CROSS_DEVICE]] %[[#ACQREL_GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_DEVICE]] %[[#SCOPE_DEVICE]] %[[#ACQREL_GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_DEVICE]] %[[#SCOPE_WORK_GROUP]] %[[#ACQREL_GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_DEVICE]] %[[#SCOPE_SUBGROUP]] %[[#ACQREL_GLOBAL]]
; CHECK-SPIRV: OpControlBarrier %[[#SCOPE_DEVICE]] %[[#SCOPE_INVOCATION]] %[[#ACQREL_GLOBAL]]

; Function Attrs: convergent norecurse nounwind
define dso_local spir_kernel void @test(ptr addrspace(1) captures(none) noundef readnone align 4 %0) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef 1, i32 noundef 1, i32 noundef 264) #2
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef 1, i32 noundef 1, i32 noundef 520) #2
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef 1, i32 noundef 1, i32 noundef 2056) #2

  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef 1, i32 noundef 0, i32 noundef 520) #2
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef 1, i32 noundef 1, i32 noundef 520) #2
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef 1, i32 noundef 2, i32 noundef 520) #2
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef 1, i32 noundef 3, i32 noundef 520) #2
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef 1, i32 noundef 4, i32 noundef 520) #2
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project 861386dbd6ff0d91636b7c674c2abb2eccd9d3f2)"}
!4 = !{i32 1}
!5 = !{!"none"}
!6 = !{!"uint*"}
!7 = !{!""}
