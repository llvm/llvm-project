; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-ERROR
; XFAIL: *

; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_masked_gather_scatter

; CHECK-DAG: OpCapability MaskedGatherScatterINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_masked_gather_scatter"

; CHECK-DAG: %[[#TYPEINT1:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#TYPEINT2:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#TYPEPTR1:]] = OpTypePointer CrossWorkgroup %[[#TYPEINT1]]
; CHECK-DAG: %[[#TYPEVEC1:]] = OpTypeVector %[[#TYPEPTR1]] 4
; CHECK-DAG: %[[#TYPEVOID:]] = OpTypeVoid
; CHECK-DAG: %[[#TYPEPTR2:]] = OpTypePointer Generic %[[#TYPEINT1]]
; CHECK-DAG: %[[#TYPEVEC2:]] = OpTypeVector %[[#TYPEPTR2]] 4
; CHECK-DAG: %[[#PTRTOVECTYPE:]] = OpTypePointer Function %[[#TYPEVEC2]]
; CHECK-DAG: %[[#TYPEPTR4:]] = OpTypePointer CrossWorkgroup %[[#TYPEINT2]]
; CHECK-DAG: %[[#TYPEVEC3:]] = OpTypeVector %[[#TYPEPTR4]] 4

; CHECK: OpVariable %[[#PTRTOVECTYPE]]
; CHECK: OpVariable %[[#PTRTOVECTYPE]]
; CHECK: OpLoad %[[#TYPEVEC2]]
; CHECK: OpStore
; CHECK: OpGenericCastToPtr %[[#TYPEVEC1]]
; CHECK: OpFunctionCall %[[#TYPEVEC3]]
; CHECK: OpInBoundsPtrAccessChain %[[#TYPEVEC3]]

define spir_kernel void @foo() {
entry:
  %arg1 = alloca <4 x ptr addrspace(4)>
  %arg2 = alloca <4 x ptr addrspace(4)>
  %0 = load <4 x ptr addrspace(4)>, ptr %arg1
  store <4 x ptr addrspace(4)> %0, ptr %arg2
  %tmp1 = addrspacecast <4 x ptr addrspace(4)> %0 to  <4 x ptr addrspace(1)>
  %tmp2 = call <4 x ptr addrspace(1)> @boo(<4 x ptr addrspace(1)> %tmp1)
  %tmp3 = getelementptr inbounds i32, <4 x ptr addrspace(1)> %tmp2, i32 1
  ret void
}

declare <4 x ptr addrspace(1)> @boo(<4 x ptr addrspace(1)> %a)
