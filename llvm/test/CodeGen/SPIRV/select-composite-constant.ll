; A `select` whose arms are composite (aggregate) constants used to crash in
; SPIRVEmitIntrinsics: preprocessCompositeConstants() rewrites the composite
; constant operands into i32 value-ids, which left the select with an aggregate
; result type but i32 operands -- an invalid state rejected by the verifier with
; "Select values must have same type as select instruction". Check that such a
; select is now lowered to a valid OpSelect over the composite type.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Two:]] = OpConstant %[[#Int]] 2
; CHECK-DAG: %[[#Array:]] = OpTypeArray %[[#Float]] %[[#Two]]
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#Float]] %[[#Float]]

; The selects must be lowered to OpSelect over the composite type (not i32).
; CHECK-DAG: %[[#ArrSel:]] = OpSelect %[[#Array]] %[[#]] %[[#]] %[[#]]
; CHECK-DAG: %[[#StructSel:]] = OpSelect %[[#Struct]] %[[#]] %[[#]] %[[#]]
; CHECK-DAG: OpCompositeExtract %[[#Float]] %[[#ArrSel]] 0
; CHECK-DAG: OpCompositeExtract %[[#Float]] %[[#ArrSel]] 1
; CHECK-DAG: OpCompositeExtract %[[#Float]] %[[#StructSel]] 0

; Array-typed composite constant (e.g. a Julia Complex{Float32}).
define spir_kernel void @select_array_constant(ptr addrspace(1) %out, i1 %c) {
  %v = select i1 %c, [2 x float] [float 1.000000e+00, float 0.000000e+00], [2 x float] zeroinitializer
  %e0 = extractvalue [2 x float] %v, 0
  %e1 = extractvalue [2 x float] %v, 1
  store float %e0, ptr addrspace(1) %out
  %p1 = getelementptr float, ptr addrspace(1) %out, i64 1
  store float %e1, ptr addrspace(1) %p1
  ret void
}

; Struct-typed composite constant (e.g. a C _Complex float).
define spir_kernel void @select_struct_constant(ptr addrspace(1) %out, i1 %c) {
  %v = select i1 %c, { float, float } { float 1.000000e+00, float 2.000000e+00 }, { float, float } zeroinitializer
  %e0 = extractvalue { float, float } %v, 0
  store float %e0, ptr addrspace(1) %out
  ret void
}
