; An aggregate member taken out with extractvalue becomes an i32 value-id when
; it feeds an aggregate PHI, select or freeze, which SPIRVEmitIntrinsics has
; already rewritten to value-ids, and it needs its composite type assigned.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Inner:]] = OpTypeStruct %[[#Float]] %[[#Float]] %[[#Float]]
; CHECK-DAG: %[[#Outer:]] = OpTypeStruct %[[#Int]] %[[#Inner]]

%Inner = type { float, float, float }
%Outer = type { i32, %Inner }

; CHECK: %[[#L0:]] = OpLoad %[[#Outer]]
; CHECK: %[[#E0:]] = OpCompositeExtract %[[#Inner]] %[[#L0]] 1
; CHECK: %[[#P:]] = OpPhi %[[#Inner]] %[[#E0]] %[[#]] %[[#]] %[[#]]
; CHECK: OpCompositeExtract %[[#Float]] %[[#P]] 0
define spir_kernel void @phi(ptr addrspace(1) %out, ptr addrspace(1) %in, i1 %c) {
entry:
  %s = load %Outer, ptr addrspace(1) %in
  %v = extractvalue %Outer %s, 1
  br i1 %c, label %zero, label %join
zero:
  br label %join
join:
  %p = phi %Inner [ %v, %entry ], [ zeroinitializer, %zero ]
  %x = extractvalue %Inner %p, 0
  store float %x, ptr addrspace(1) %out
  ret void
}

; CHECK: %[[#L1:]] = OpLoad %[[#Outer]]
; CHECK: %[[#E1:]] = OpCompositeExtract %[[#Inner]] %[[#L1]] 1
; CHECK: %[[#S:]] = OpSelect %[[#Inner]] %[[#]] %[[#E1]] %[[#]]
; CHECK: OpCompositeExtract %[[#Float]] %[[#S]] 1
define spir_kernel void @select(ptr addrspace(1) %out, ptr addrspace(1) %in, i1 %c) {
  %s = load %Outer, ptr addrspace(1) %in
  %v = extractvalue %Outer %s, 1
  %p = select i1 %c, %Inner %v, %Inner zeroinitializer
  %x = extractvalue %Inner %p, 1
  store float %x, ptr addrspace(1) %out
  ret void
}

; CHECK: %[[#L2:]] = OpLoad %[[#Outer]]
; CHECK: %[[#E2:]] = OpCompositeExtract %[[#Inner]] %[[#L2]] 1
; CHECK: OpCompositeExtract %[[#Float]] %[[#]] 2
define spir_kernel void @freeze(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %s = load %Outer, ptr addrspace(1) %in
  %v = extractvalue %Outer %s, 1
  %f = freeze %Inner %v
  %x = extractvalue %Inner %f, 2
  store float %x, ptr addrspace(1) %out
  ret void
}
