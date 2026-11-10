; Adapted from Khronos Translator test suite: test/extensions/INTEL/SPV_INTEL_device_barrier/

;; void helper(int scope)
;; {
;;    __spirv_ControlBarrier(scope, 1, 520);
;; }
;;
;; kernel void test(global uint* dst)
;; {
;;    int scope = magic_get_scope();
;;    __spirv_ControlBarrier(scope, 1, 264);  // local
;;    __spirv_ControlBarrier(scope, 1, 520);  // global
;;    __spirv_ControlBarrier(scope, 1, 2056); // image
;;
;;    __spirv_ControlBarrier(scope, 0, 520);  // global, all_svm_devices
;;    __spirv_ControlBarrier(scope, 1, 520);  // global, device
;;    __spirv_ControlBarrier(scope, 2, 520);  // global, work_group
;;    __spirv_ControlBarrier(scope, 3, 520);  // global, subgroup
;;    __spirv_ControlBarrier(scope, 4, 520);  // global, work_item
;;
;;    helper(scope);
;;}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_barrier %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_barrier %s -o - -filetype=obj | spirv-val %}

; Without the extension, a runtime execution scope does not cause the
; extension or capability to be declared.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-DISABLED

; CHECK: OpCapability DeviceBarrierINTEL
; CHECK: OpExtension "SPV_INTEL_device_barrier"
; CHECK: %[[#UINT:]] = OpTypeInt 32 0
;
; Scopes:
; CHECK-DAG: %[[#SCOPE_CROSS_DEVICE:]] = OpConstantNull %[[#UINT]]
; CHECK-DAG: %[[#SCOPE_DEVICE:]] = OpConstant %[[#UINT]] 1{{$}}
; CHECK-DAG: %[[#SCOPE_WORK_GROUP:]] = OpConstant %[[#UINT]] 2{{$}}
; CHECK-DAG: %[[#SCOPE_SUBGROUP:]] = OpConstant %[[#UINT]] 3{{$}}
; CHECK-DAG: %[[#SCOPE_INVOCATION:]] = OpConstant %[[#UINT]] 4{{$}}
;
; Memory Semantics:
; 0x8 AcquireRelease + 0x100 WorkgroupMemory
; CHECK-DAG: %[[#ACQREL_LOCAL:]] = OpConstant %[[#UINT]] 264
; 0x8 AcquireRelease + 0x200 CrossWorkgroupMemory
; CHECK-DAG: %[[#ACQREL_GLOBAL:]] = OpConstant %[[#UINT]] 520
; 0x8 AcquireRelease + 0x800 ImageMemory
; CHECK-DAG: %[[#ACQREL_IMAGE:]] = OpConstant %[[#UINT]] 2056
;
; Runtime execution scope passed as a function parameter:
; CHECK: %[[#PARAM:]] = OpFunctionParameter %[[#UINT]]
; CHECK: OpControlBarrier %[[#PARAM]] %[[#SCOPE_DEVICE]] %[[#ACQREL_GLOBAL]]
;
; Runtime execution scope returned from a function call:
; CHECK: %[[#EXEC_SCOPE:]] = OpFunctionCall %[[#UINT]]
;
; CHECK: OpControlBarrier %[[#EXEC_SCOPE]] %[[#SCOPE_DEVICE]] %[[#ACQREL_LOCAL]]
; CHECK: OpControlBarrier %[[#EXEC_SCOPE]] %[[#SCOPE_DEVICE]] %[[#ACQREL_GLOBAL]]
; CHECK: OpControlBarrier %[[#EXEC_SCOPE]] %[[#SCOPE_DEVICE]] %[[#ACQREL_IMAGE]]
;
; CHECK: OpControlBarrier %[[#EXEC_SCOPE]] %[[#SCOPE_CROSS_DEVICE]] %[[#ACQREL_GLOBAL]]
; CHECK: OpControlBarrier %[[#EXEC_SCOPE]] %[[#SCOPE_DEVICE]] %[[#ACQREL_GLOBAL]]
; CHECK: OpControlBarrier %[[#EXEC_SCOPE]] %[[#SCOPE_WORK_GROUP]] %[[#ACQREL_GLOBAL]]
; CHECK: OpControlBarrier %[[#EXEC_SCOPE]] %[[#SCOPE_SUBGROUP]] %[[#ACQREL_GLOBAL]]
; CHECK: OpControlBarrier %[[#EXEC_SCOPE]] %[[#SCOPE_INVOCATION]] %[[#ACQREL_GLOBAL]]

; CHECK-DISABLED-NOT: OpCapability DeviceBarrierINTEL
; CHECK-DISABLED-NOT: OpExtension "SPV_INTEL_device_barrier"
; CHECK-DISABLED: OpControlBarrier

define spir_func void @helper(i32 %scope) {
  call spir_func void @_Z22__spirv_ControlBarrieriii(i32 %scope, i32 1, i32 520)
  ret void
}

define dso_local spir_kernel void @test(ptr addrspace(1) noundef align 4 %0) {
  %2 = call noundef i32 @magic_get_scope()
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef %2, i32 noundef 1, i32 noundef 264)
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef %2, i32 noundef 1, i32 noundef 520)
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef %2, i32 noundef 1, i32 noundef 2056)

  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef %2, i32 noundef 0, i32 noundef 520)
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef %2, i32 noundef 1, i32 noundef 520)
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef %2, i32 noundef 2, i32 noundef 520)
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef %2, i32 noundef 3, i32 noundef 520)
  tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef %2, i32 noundef 4, i32 noundef 520)

  call spir_func void @helper(i32 %2)
  ret void
}

declare dso_local spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef, i32 noundef, i32 noundef)

declare spir_func i32 @magic_get_scope()
