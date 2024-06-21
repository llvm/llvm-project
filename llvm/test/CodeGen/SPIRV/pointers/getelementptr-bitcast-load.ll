; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#VEC3:]] = OpTypeVector %[[#INT8]] 3
; CHECK-DAG: %[[#VEC4:]] = OpTypeVector %[[#INT8]] 4
; CHECK-DAG: %[[#PTR_VEC3:]] = OpTypePointer CrossWorkgroup %[[#VEC3]]
; CHECK-DAG: %[[#PTR_VEC4:]] = OpTypePointer CrossWorkgroup %[[#VEC4]]

; CHECK: %[[#AC1:]] = OpInBoundsPtrAccessChain %[[#PTR_VEC3]] %[[#]] %[[#]]
; CHECK: %[[#BC1:]] = OpBitcast %[[#PTR_VEC4]] %[[#AC1]]
; CHECK: %[[#LD1:]] = OpLoad %[[#VEC4]] %[[#BC1]] Aligned 4
; CHECK: OpReturn

define spir_kernel void @foo(ptr addrspace(1) %a, i64 %b) {
  %index = getelementptr inbounds <3 x i8>, ptr addrspace(1) %a, i64 %b
  %loadv = load <4 x i8>, ptr addrspace(1) %index, align 4
  ret void
}

; CHECK: %[[#AC2:]] = OpInBoundsPtrAccessChain %[[#PTR_VEC3]] %[[#]] %[[#]]
; CHECK: %[[#BC2:]] = OpBitcast %[[#PTR_VEC4]] %[[#AC2]]
; CHECK: %[[#LD2:]] = OpLoad %[[#VEC4]] %[[#BC2]] Aligned 4
; CHECK: OpReturn

define spir_kernel void @bar(ptr addrspace(1) %a, i64 %b) {
  %index = getelementptr inbounds <3 x i8>, ptr addrspace(1) %a, i64 %b
; This redundant bitcast is left here itentionally to simulate the conversion
; from older LLVM IR with typed pointers.
  %cast = bitcast ptr addrspace(1) %index to ptr addrspace(1)
  %loadv = load <4 x i8>, ptr addrspace(1) %cast, align 4
  ret void
}
