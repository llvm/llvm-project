; A `select` between aggregate (array/struct) values whose arms are not both
; composite constants -- one or both arms are loaded, or are themselves a
; select -- used to crash in SPIRVEmitIntrinsics ("illegal aggregate intrinsic
; user", or the verifier's "Select values must have same type as select
; instruction"). Check that such selects are lowered to a valid OpSelect over
; the composite type. The all-constant case is covered by
; select-composite-constant.ll.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Two:]] = OpConstant %[[#Int]] 2
; CHECK-DAG: %[[#Array:]] = OpTypeArray %[[#Float]] %[[#Two]]
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#Float]] %[[#Float]]

; Both arms are loaded (non-constant) aggregates.
; CHECK: %[[#A:]] = OpLoad %[[#Array]]
; CHECK: %[[#B:]] = OpLoad %[[#Array]]
; CHECK: %[[#Sel0:]] = OpSelect %[[#Array]] %[[#]] %[[#A]] %[[#B]]
; CHECK: OpCompositeExtract %[[#Float]] %[[#Sel0]] 0
; CHECK: OpCompositeExtract %[[#Float]] %[[#Sel0]] 1
define spir_kernel void @both_loaded(ptr addrspace(1) %out, ptr addrspace(1) %pa, ptr addrspace(1) %pb, i1 %c) {
  %a = load [2 x float], ptr addrspace(1) %pa
  %b = load [2 x float], ptr addrspace(1) %pb
  %v = select i1 %c, [2 x float] %a, [2 x float] %b
  %e0 = extractvalue [2 x float] %v, 0
  %e1 = extractvalue [2 x float] %v, 1
  store float %e0, ptr addrspace(1) %out
  %p1 = getelementptr float, ptr addrspace(1) %out, i64 1
  store float %e1, ptr addrspace(1) %p1
  ret void
}

; Mixed: one arm loaded, the other a composite constant.
; CHECK: %[[#M:]] = OpLoad %[[#Array]]
; CHECK: %[[#Sel1:]] = OpSelect %[[#Array]] %[[#]] %[[#M]] %[[#]]
; CHECK: OpCompositeExtract %[[#Float]] %[[#Sel1]] 0
define spir_kernel void @mixed_array(ptr addrspace(1) %out, ptr addrspace(1) %pa, i1 %c) {
  %a = load [2 x float], ptr addrspace(1) %pa
  %v = select i1 %c, [2 x float] %a, [2 x float] [float 1.000000e+00, float 0.000000e+00]
  %e0 = extractvalue [2 x float] %v, 0
  store float %e0, ptr addrspace(1) %out
  ret void
}

; Mixed struct-typed select.
; CHECK: %[[#S:]] = OpLoad %[[#Struct]]
; CHECK: %[[#Sel2:]] = OpSelect %[[#Struct]] %[[#]] %[[#S]] %[[#]]
; CHECK: OpCompositeExtract %[[#Float]] %[[#Sel2]] 0
define spir_kernel void @mixed_struct(ptr addrspace(1) %out, ptr addrspace(1) %pa, i1 %c) {
  %a = load { float, float }, ptr addrspace(1) %pa
  %v = select i1 %c, { float, float } %a, { float, float } { float 1.000000e+00, float 2.000000e+00 }
  %e0 = extractvalue { float, float } %v, 0
  store float %e0, ptr addrspace(1) %out
  ret void
}

; A select whose arm is itself a select.
; CHECK: %[[#Inner:]] = OpSelect %[[#Array]] %[[#]] %[[#]] %[[#]]
; CHECK: %[[#Outer:]] = OpSelect %[[#Array]] %[[#]] %[[#Inner]] %[[#]]
; CHECK: OpCompositeExtract %[[#Float]] %[[#Outer]] 0
define spir_kernel void @nested(ptr addrspace(1) %out, ptr addrspace(1) %pa, i1 %c, i1 %d) {
  %a = load [2 x float], ptr addrspace(1) %pa
  %inner = select i1 %d, [2 x float] %a, [2 x float] zeroinitializer
  %v = select i1 %c, [2 x float] %inner, [2 x float] [float 1.000000e+00, float 0.000000e+00]
  %e0 = extractvalue [2 x float] %v, 0
  store float %e0, ptr addrspace(1) %out
  ret void
}

; The aggregate select result is stored directly, without an extractvalue.
; CHECK: %[[#Sel3:]] = OpSelect %[[#Array]] %[[#]] %[[#]] %[[#]]
; CHECK: OpStore %[[#]] %[[#Sel3]]
define spir_kernel void @store_direct(ptr addrspace(1) %out, ptr addrspace(1) %pa, i1 %c) {
  %a = load [2 x float], ptr addrspace(1) %pa
  %v = select i1 %c, [2 x float] %a, [2 x float] zeroinitializer
  store [2 x float] %v, ptr addrspace(1) %out
  ret void
}
