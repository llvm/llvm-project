; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CORE
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s --check-prefixes=CHECK,EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; One-element vectors are normalized only without the extension. Other
; unsupported ranks must retain the byte-pointer fallback in declarations.
; CHECK-DAG: OpName %[[#Single:]] "_Z6singlePDv1_f"
; CHECK-DAG: OpName %[[#Five:]] "_Z4fivePDv5_f"
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CORE-DAG: %[[#Char:]] = OpTypeInt 8 0
; CORE-DAG: %[[#SinglePtr:]] = OpTypePointer CrossWorkgroup %[[#Float]]
; CORE-DAG: %[[#FivePtr:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; EXT-DAG: %[[#Int:]] = OpTypeInt 32 0
; EXT-DAG: %[[#One:]] = OpConstant %[[#Int]] 1
; EXT-DAG: %[[#FiveCount:]] = OpConstant %[[#Int]] 5
; EXT-DAG: %[[#Vector1:]] = OpTypeVectorIdEXT %[[#Float]] %[[#One]]
; EXT-DAG: %[[#Vector5:]] = OpTypeVectorIdEXT %[[#Float]] %[[#FiveCount]]
; EXT-DAG: %[[#SinglePtr:]] = OpTypePointer CrossWorkgroup %[[#Vector1]]
; EXT-DAG: %[[#FivePtr:]] = OpTypePointer CrossWorkgroup %[[#Vector5]]
; CHECK-DAG: %[[#SingleTy:]] = OpTypeFunction %[[#Void]] %[[#SinglePtr]]
; CHECK-DAG: %[[#FiveTy:]] = OpTypeFunction %[[#Void]] %[[#FivePtr]]
; CHECK: %[[#Single]] = OpFunction %[[#Void]] None %[[#SingleTy]]
; CHECK-NEXT: OpFunctionParameter %[[#SinglePtr]]
; CHECK: %[[#Five]] = OpFunction %[[#Void]] None %[[#FiveTy]]
; CHECK-NEXT: OpFunctionParameter %[[#FivePtr]]

declare spir_func void @_Z6singlePDv1_f(ptr addrspace(1))
declare spir_func void @_Z4fivePDv5_f(ptr addrspace(1))

define spir_kernel void @test(ptr addrspace(1) %p) {
  call spir_func void @_Z6singlePDv1_f(ptr addrspace(1) %p)
  call spir_func void @_Z4fivePDv5_f(ptr addrspace(1) %p)
  ret void
}
