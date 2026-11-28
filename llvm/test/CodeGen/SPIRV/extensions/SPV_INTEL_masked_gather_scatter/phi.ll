; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; A PHI of a vector of pointers is legal with SPV_INTEL_masked_gather_scatter.
; CHECK: OpCapability MaskedGatherScatterINTEL
; CHECK: OpExtension "SPV_INTEL_masked_gather_scatter"
; CHECK-DAG: [[PTR:%.+]] = OpTypePointer Generic {{%.+}}
; CHECK-DAG: [[V4PTR:%.+]] = OpTypeVector [[PTR]] 4

; CHECK: OpFunction [[V4PTR]]
; CHECK: [[A:%.+]] = OpFunctionParameter [[V4PTR]]
; CHECK: [[B:%.+]] = OpFunctionParameter [[V4PTR]]
; CHECK: [[P:%.+]] = OpPhi [[V4PTR]] [[A]] {{%.+}} [[B]] {{%.+}}
; CHECK: OpReturnValue [[P]]
define <4 x ptr addrspace(4)> @phi_ptrv4(i1 %c, <4 x ptr addrspace(4)> %a, <4 x ptr addrspace(4)> %b) {
entry:
  br i1 %c, label %t, label %e
t:
  br label %e
e:
  %p = phi <4 x ptr addrspace(4)> [ %a, %entry ], [ %b, %t ]
  ret <4 x ptr addrspace(4)> %p
}
