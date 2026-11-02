; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.3-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s --check-prefixes=CHECK,SPV13
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.3-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val --target-env spv1.3 %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64v1.3-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s --check-prefixes=CHECK,SPV13
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.3-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val --target-env spv1.3 %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.4-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s --check-prefixes=CHECK,SPV14
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.4-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64v1.4-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s --check-prefixes=CHECK,SPV14
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.4-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; Vectors of pointers require SPV_INTEL_masked_gather_scatter.
; CHECK: OpCapability MaskedGatherScatterINTEL
; CHECK: OpExtension "SPV_INTEL_masked_gather_scatter"
; CHECK-DAG: [[BOOL:%.+]] = OpTypeBool
; CHECK-DAG: [[V2BOOL:%.+]] = OpTypeVector [[BOOL]] 2
; CHECK-DAG: [[PTR:%.+]] = OpTypePointer CrossWorkgroup {{%.+}}
; CHECK-DAG: [[V2PTR:%.+]] = OpTypeVector [[PTR]] 2

; CHECK: OpFunction [[V2PTR]]
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter [[BOOL]]
; CHECK-NEXT: [[T:%.+]] = OpFunctionParameter [[V2PTR]]
; CHECK-NEXT: [[F:%.+]] = OpFunctionParameter [[V2PTR]]
; CHECK: OpLabel
; SPV13: [[SPLAT:%.+]] = OpCompositeConstruct [[V2BOOL]] [[C]] [[C]]
; SPV13-NEXT: [[R:%.+]] = OpSelect [[V2PTR]] [[SPLAT]] [[T]] [[F]]
; SPV14-NOT: OpCompositeConstruct
; SPV14: [[R:%.+]] = OpSelect [[V2PTR]] [[C]] [[T]] [[F]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x ptr addrspace(1)> @select_ptrv2(i1 %c, <2 x ptr addrspace(1)> %t, <2 x ptr addrspace(1)> %f) {
  %r = select i1 %c, <2 x ptr addrspace(1)> %t, <2 x ptr addrspace(1)> %f
  ret <2 x ptr addrspace(1)> %r
}

; CHECK: OpFunction [[V2PTR]]
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter [[V2BOOL]]
; CHECK-NEXT: [[T:%.+]] = OpFunctionParameter [[V2PTR]]
; CHECK-NEXT: [[F:%.+]] = OpFunctionParameter [[V2PTR]]
; CHECK: OpLabel
; CHECK-NOT: OpCompositeConstruct
; CHECK: [[R:%.+]] = OpSelect [[V2PTR]] [[C]] [[T]] [[F]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <2 x ptr addrspace(1)> @select_v2ptrv2(<2 x i1> %c, <2 x ptr addrspace(1)> %t, <2 x ptr addrspace(1)> %f) {
  %r = select <2 x i1> %c, <2 x ptr addrspace(1)> %t, <2 x ptr addrspace(1)> %f
  ret <2 x ptr addrspace(1)> %r
}
