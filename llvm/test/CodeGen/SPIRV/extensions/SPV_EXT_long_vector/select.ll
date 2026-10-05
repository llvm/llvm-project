; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.3-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s --check-prefixes=CHECK,SPV13
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.3-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val --target-env spv1.3 %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64v1.3-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s --check-prefixes=CHECK,SPV13
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.3-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val --target-env spv1.3 %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.4-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s --check-prefixes=CHECK,SPV14
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.4-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64v1.4-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s --check-prefixes=CHECK,SPV14
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.4-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; The component count of OpTypeVectorIdEXT is an ID, not a literal.
; Before SPIR-V 1.4, splat the scalar condition to a matching boolean vector.
; CHECK: OpCapability LongVectorEXT
; CHECK: OpExtension "SPV_EXT_long_vector"
; CHECK-DAG: [[BOOL:%.+]] = OpTypeBool
; CHECK-DAG: [[INT:%.+]] = OpTypeInt 32 0
; CHECK-DAG: [[FIVE:%.+]] = OpConstant [[INT]] 5
; CHECK-DAG: [[V5INT:%.+]] = OpTypeVectorIdEXT [[INT]] [[FIVE]]
; SPV13-DAG: [[V5BOOL:%.+]] = OpTypeVectorIdEXT [[BOOL]] [[FIVE]]

; CHECK: OpFunction [[V5INT]]
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter [[BOOL]]
; CHECK-NEXT: [[T:%.+]] = OpFunctionParameter [[V5INT]]
; CHECK-NEXT: [[F:%.+]] = OpFunctionParameter [[V5INT]]
; CHECK: OpLabel
; SPV13: [[SPLAT:%.+]] = OpCompositeConstruct [[V5BOOL]] [[C]] [[C]] [[C]] [[C]] [[C]]
; SPV13-NEXT: [[R:%.+]] = OpSelect [[V5INT]] [[SPLAT]] [[T]] [[F]]
; SPV14-NOT: OpCompositeConstruct
; SPV14: [[R:%.+]] = OpSelect [[V5INT]] [[C]] [[T]] [[F]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <5 x i32> @select_i32v5(i1 %c, <5 x i32> %t, <5 x i32> %f) {
  %r = select i1 %c, <5 x i32> %t, <5 x i32> %f
  ret <5 x i32> %r
}
