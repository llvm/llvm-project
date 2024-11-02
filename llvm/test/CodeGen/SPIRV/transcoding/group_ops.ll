; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[#ScopeCrossWorkgroup:]] = OpConstant %[[#int]] 0
; CHECK-SPIRV-DAG: %[[#ScopeWorkgroup:]] = OpConstant %[[#int]] 2
; CHECK-SPIRV-DAG: %[[#ScopeSubgroup:]] = OpConstant %[[#int]] 3

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupFMax %[[#float]] %[[#ScopeWorkgroup]] Reduce
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupFMax(float a, global float *res) {
;;   res[0] = work_group_reduce_max(a);
;; }

define dso_local spir_kernel void @testWorkGroupFMax(float noundef %a, float addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func float @_Z21work_group_reduce_maxf(float noundef %a)
  store float %call, float addrspace(1)* %res, align 4
  ret void
}

declare spir_func float @_Z21work_group_reduce_maxf(float noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupFMin %[[#float]] %[[#ScopeWorkgroup]] Reduce
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupFMin(float a, global float *res) {
;;   res[0] = work_group_reduce_min(a);
;; }

define dso_local spir_kernel void @testWorkGroupFMin(float noundef %a, float addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func float @_Z21work_group_reduce_minf(float noundef %a)
  store float %call, float addrspace(1)* %res, align 4
  ret void
}

declare spir_func float @_Z21work_group_reduce_minf(float noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupFAdd %[[#float]] %[[#ScopeWorkgroup]] Reduce
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupFAdd(float a, global float *res) {
;;   res[0] = work_group_reduce_add(a);
;; }

define dso_local spir_kernel void @testWorkGroupFAdd(float noundef %a, float addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func float @_Z21work_group_reduce_addf(float noundef %a)
  store float %call, float addrspace(1)* %res, align 4
  ret void
}

declare spir_func float @_Z21work_group_reduce_addf(float noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupFMax %[[#float]] %[[#ScopeWorkgroup]] InclusiveScan
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupScanInclusiveFMax(float a, global float *res) {
;;   res[0] = work_group_scan_inclusive_max(a);
;; }

define dso_local spir_kernel void @testWorkGroupScanInclusiveFMax(float noundef %a, float addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func float @_Z29work_group_scan_inclusive_maxf(float noundef %a)
  store float %call, float addrspace(1)* %res, align 4
  ret void
}

declare spir_func float @_Z29work_group_scan_inclusive_maxf(float noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupFMax %[[#float]] %[[#ScopeWorkgroup]] ExclusiveScan
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupScanExclusiveFMax(float a, global float *res) {
;;   res[0] = work_group_scan_exclusive_max(a);
;; }

define dso_local spir_kernel void @testWorkGroupScanExclusiveFMax(float noundef %a, float addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func float @_Z29work_group_scan_exclusive_maxf(float noundef %a)
  store float %call, float addrspace(1)* %res, align 4
  ret void
}

declare spir_func float @_Z29work_group_scan_exclusive_maxf(float noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupSMax %[[#int]] %[[#ScopeWorkgroup]] Reduce
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupSMax(int a, global int *res) {
;;   res[0] = work_group_reduce_max(a);
;; }

define dso_local spir_kernel void @testWorkGroupSMax(i32 noundef %a, i32 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func i32 @_Z21work_group_reduce_maxi(i32 noundef %a)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare spir_func i32 @_Z21work_group_reduce_maxi(i32 noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupSMin %[[#int]] %[[#ScopeWorkgroup]] Reduce
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupSMin(int a, global int *res) {
;;   res[0] = work_group_reduce_min(a);
;; }

define dso_local spir_kernel void @testWorkGroupSMin(i32 noundef %a, i32 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func i32 @_Z21work_group_reduce_mini(i32 noundef %a)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare spir_func i32 @_Z21work_group_reduce_mini(i32 noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupIAdd %[[#int]] %[[#ScopeWorkgroup]] Reduce
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupIAddSigned(int a, global int *res) {
;;   res[0] = work_group_reduce_add(a);
;; }

define dso_local spir_kernel void @testWorkGroupIAddSigned(i32 noundef %a, i32 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func i32 @_Z21work_group_reduce_addi(i32 noundef %a)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare spir_func i32 @_Z21work_group_reduce_addi(i32 noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupIAdd %[[#int]] %[[#ScopeWorkgroup]] Reduce
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupIAddUnsigned(uint a, global uint *res) {
;;   res[0] = work_group_reduce_add(a);
;; }

define dso_local spir_kernel void @testWorkGroupIAddUnsigned(i32 noundef %a, i32 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func i32 @_Z21work_group_reduce_addj(i32 noundef %a)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare spir_func i32 @_Z21work_group_reduce_addj(i32 noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupUMax %[[#int]] %[[#ScopeWorkgroup]] Reduce
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupUMax(uint a, global uint *res) {
;;   res[0] = work_group_reduce_max(a);
;; }

define dso_local spir_kernel void @testWorkGroupUMax(i32 noundef %a, i32 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func i32 @_Z21work_group_reduce_maxj(i32 noundef %a)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare spir_func i32 @_Z21work_group_reduce_maxj(i32 noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupUMax %[[#int]] %[[#ScopeSubgroup]] Reduce
; CHECK-SPIRV: OpFunctionEnd

;; #pragma OPENCL EXTENSION cl_khr_subgroups: enable
;; kernel void testSubGroupUMax(uint a, global uint *res) {
;;   res[0] = sub_group_reduce_max(a);
;; }
;; #pragma OPENCL EXTENSION cl_khr_subgroups: disable

define dso_local spir_kernel void @testSubGroupUMax(i32 noundef %a, i32 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func i32 @_Z20sub_group_reduce_maxj(i32 noundef %a)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare spir_func i32 @_Z20sub_group_reduce_maxj(i32 noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupUMax %[[#int]] %[[#ScopeWorkgroup]] InclusiveScan
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupScanInclusiveUMax(uint a, global uint *res) {
;;   res[0] = work_group_scan_inclusive_max(a);
;; }

define dso_local spir_kernel void @testWorkGroupScanInclusiveUMax(i32 noundef %a, i32 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func i32 @_Z29work_group_scan_inclusive_maxj(i32 noundef %a)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare spir_func i32 @_Z29work_group_scan_inclusive_maxj(i32 noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupUMax %[[#int]] %[[#ScopeWorkgroup]] ExclusiveScan
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupScanExclusiveUMax(uint a, global uint *res) {
;;   res[0] = work_group_scan_exclusive_max(a);
;; }

define dso_local spir_kernel void @testWorkGroupScanExclusiveUMax(i32 noundef %a, i32 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func i32 @_Z29work_group_scan_exclusive_maxj(i32 noundef %a)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare spir_func i32 @_Z29work_group_scan_exclusive_maxj(i32 noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupUMin %[[#int]] %[[#ScopeWorkgroup]] Reduce
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupUMin(uint a, global uint *res) {
;;   res[0] = work_group_reduce_min(a);
;; }

define dso_local spir_kernel void @testWorkGroupUMin(i32 noundef %a, i32 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %call = call spir_func i32 @_Z21work_group_reduce_minj(i32 noundef %a)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare spir_func i32 @_Z21work_group_reduce_minj(i32 noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupBroadcast %[[#int]] %[[#ScopeWorkgroup]] %[[#BroadcastValue:]] %[[#BroadcastLocalId:]]
; CHECK-SPIRV: %[[#]] = OpGroupBroadcast %[[#int]] %[[#ScopeCrossWorkgroup]] %[[#BroadcastValue]] %[[#BroadcastLocalId]]
; CHECK-SPIRV: OpFunctionEnd

;; kernel void testWorkGroupBroadcast(uint a, global size_t *id, global int *res) {
;;   res[0] = work_group_broadcast(a, *id);
;; }

define dso_local spir_kernel void @testWorkGroupBroadcast(i32 noundef %a, i32 addrspace(1)* nocapture noundef readonly %id, i32 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %0 = load i32, i32 addrspace(1)* %id, align 4
  %call = call spir_func i32 @_Z20work_group_broadcastjj(i32 noundef %a, i32 noundef %0)
  store i32 %call, i32 addrspace(1)* %res, align 4
  %call1 = call spir_func i32 @__spirv_GroupBroadcast(i32 0, i32 noundef %a, i32 noundef %0)
  ret void
}

declare spir_func i32 @_Z20work_group_broadcastjj(i32 noundef, i32 noundef) local_unnamed_addr
declare spir_func i32 @__spirv_GroupBroadcast(i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupFAdd %[[#float]] %[[#ScopeCrossWorkgroup]] Reduce %[[#FValue:]]
; CHECK-SPIRV: %[[#]] = OpGroupFMin %[[#float]] %[[#ScopeWorkgroup]] InclusiveScan %[[#FValue]]
; CHECK-SPIRV: %[[#]] = OpGroupFMax %[[#float]] %[[#ScopeSubgroup]] ExclusiveScan %[[#FValue]]
; CHECK-SPIRV: %[[#]] = OpGroupIAdd %[[#int]] %[[#ScopeCrossWorkgroup]] Reduce %[[#IValue:]]
; CHECK-SPIRV: %[[#]] = OpGroupUMin %[[#int]] %[[#ScopeWorkgroup]] InclusiveScan %[[#IValue]]
; CHECK-SPIRV: %[[#]] = OpGroupSMin %[[#int]] %[[#ScopeSubgroup]] ExclusiveScan %[[#IValue]]
; CHECK-SPIRV: %[[#]] = OpGroupUMax %[[#int]] %[[#ScopeCrossWorkgroup]] Reduce %[[#IValue]]
; CHECK-SPIRV: %[[#]] = OpGroupSMax %[[#int]] %[[#ScopeWorkgroup]] InclusiveScan %[[#IValue]]
; CHECK-SPIRV: OpFunctionEnd

define spir_kernel void @foo(float %a, i32 %b) {
entry:
  %f1 = call spir_func float @__spirv_GroupFAdd(i32 0, i32 0, float %a)
  %f2 = call spir_func float @__spirv_GroupFMin(i32 2, i32 1, float %a)
  %f3 = call spir_func float @__spirv_GroupFMax(i32 3, i32 2, float %a)
  %i1 = call spir_func i32 @__spirv_GroupIAdd(i32 0, i32 0, i32 %b)
  %i2 = call spir_func i32 @__spirv_GroupUMin(i32 2, i32 1, i32 %b)
  %i3 = call spir_func i32 @__spirv_GroupSMin(i32 3, i32 2, i32 %b)
  %i4 = call spir_func i32 @__spirv_GroupUMax(i32 0, i32 0, i32 %b)
  %i5 = call spir_func i32 @__spirv_GroupSMax(i32 2, i32 1, i32 %b)
  ret void
}

declare spir_func float @__spirv_GroupFAdd(i32, i32, float)
declare spir_func float @__spirv_GroupFMin(i32, i32, float)
declare spir_func float @__spirv_GroupFMax(i32, i32, float)
declare spir_func i32 @__spirv_GroupIAdd(i32, i32, i32)
declare spir_func i32 @__spirv_GroupUMin(i32, i32, i32)
declare spir_func i32 @__spirv_GroupSMin(i32, i32, i32)
declare spir_func i32 @__spirv_GroupUMax(i32, i32, i32)
declare spir_func i32 @__spirv_GroupSMax(i32, i32, i32)
