; A pointer in a vector member of a constant arm is bitcast to the vector's
; element type; the vector and struct are rebuilt with OpCompositeInsert.

; RUN: llc -O0 --spirv-ext=+SPV_INTEL_masked_gather_scatter -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_INTEL_masked_gather_scatter -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#P8:]] = OpTypePointer CrossWorkgroup %[[#]]
; CHECK-DAG: %[[#V:]] = OpTypeVector %[[#P8]] 2
; CHECK-DAG: %[[#S:]] = OpTypeStruct %[[#V]] %[[#]]

; CHECK: OpFunction %[[#S]]
; CHECK: %[[#C:]] = OpBitcast %[[#P8]] %[[#]]
; CHECK: %[[#V0:]] = OpCompositeInsert %[[#V]] %[[#C]] %[[#]] 0
; CHECK: %[[#V1:]] = OpCompositeInsert %[[#V]] %[[#]] %[[#V0]] 1
; CHECK: %[[#S0:]] = OpCompositeInsert %[[#S]] %[[#V1]] %[[#]] 0
; CHECK: %[[#S1:]] = OpCompositeInsert %[[#S]] %[[#]] %[[#S0]] 1
; CHECK: OpSelect %[[#S]] %[[#]] %[[#]] %[[#S1]]

; The element type comes from @g after a null too.
; CHECK: OpFunction %[[#S]]
; CHECK: %[[#NV0:]] = OpCompositeInsert %[[#V]] %[[#]] %[[#]] 0
; CHECK: %[[#NC:]] = OpBitcast %[[#P8]] %[[#]]
; CHECK: %[[#NV1:]] = OpCompositeInsert %[[#V]] %[[#NC]] %[[#NV0]] 1
; CHECK: %[[#NS0:]] = OpCompositeInsert %[[#S]] %[[#NV1]] %[[#]] 0
; CHECK: %[[#NS1:]] = OpCompositeInsert %[[#S]] %[[#]] %[[#NS0]] 1
; CHECK: OpSelect %[[#S]] %[[#]] %[[#]] %[[#NS1]]

; CHECK: OpFunction %[[#S]]
; CHECK: %[[#PV1:]] = OpCompositeInsert %[[#V]] %[[#]] %[[#]] 1
; CHECK: %[[#PS0:]] = OpCompositeInsert %[[#S]] %[[#PV1]] %[[#]] 0
; CHECK: %[[#PS1:]] = OpCompositeInsert %[[#S]] %[[#]] %[[#PS0]] 1
; CHECK: OpPhi %[[#S]] %[[#]] %[[#]] %[[#PS1]] %[[#]]

@g = addrspace(1) global i32 0

%S = type { <2 x ptr addrspace(1)>, i32 }

define %S @select(i1 %c) {
  %s = select i1 %c, %S zeroinitializer, %S { <2 x ptr addrspace(1)> <ptr addrspace(1) @g, ptr addrspace(1) null>, i32 1 }
  ret %S %s
}

define %S @select_null_first(i1 %c) {
  %s = select i1 %c, %S zeroinitializer, %S { <2 x ptr addrspace(1)> <ptr addrspace(1) null, ptr addrspace(1) @g>, i32 1 }
  ret %S %s
}

define %S @phi(i1 %c) {
entry:
  br i1 %c, label %then, label %join
then:
  br label %join
join:
  %p = phi %S [ zeroinitializer, %entry ], [ { <2 x ptr addrspace(1)> <ptr addrspace(1) @g, ptr addrspace(1) @g>, i32 1 }, %then ]
  ret %S %p
}
