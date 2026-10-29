; A multi-register aggregate arm of a select (here from llvm.sincos and
; llvm.frexp) is rebuilt with insertvalue. The rebuilt chain must start from a
; typed placeholder: starting from a bare poison, every such chain shared one
; uniqued i32 value, so the second struct got the first one's type.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Pair:]] = OpTypeStruct %[[#Float]] %[[#Float]]
; CHECK-DAG: %[[#Frexp:]] = OpTypeStruct %[[#Float]] %[[#Int]]
; CHECK-DAG: %[[#PairInit:]] = OpUndef %[[#Pair]]
; CHECK-DAG: %[[#FrexpInit:]] = OpUndef %[[#Frexp]]
; CHECK: OpCompositeInsert %[[#Pair]] %[[#]] %[[#PairInit]] 0
; CHECK: OpSelect %[[#Pair]]
; CHECK: OpCompositeInsert %[[#Frexp]] %[[#]] %[[#FrexpInit]] 0
; CHECK: OpSelect %[[#Frexp]]

define spir_kernel void @k(ptr addrspace(1) %out, float %a, i1 %c) {
  %sc = call { float, float } @llvm.sincos.f32(float %a)
  %s1 = select i1 %c, { float, float } zeroinitializer, { float, float } %sc
  %x = extractvalue { float, float } %s1, 0
  store float %x, ptr addrspace(1) %out
  %fe = call { float, i32 } @llvm.frexp.f32.i32(float %a)
  %s2 = select i1 %c, { float, i32 } zeroinitializer, { float, i32 } %fe
  %e = extractvalue { float, i32 } %s2, 1
  %p = getelementptr inbounds i8, ptr addrspace(1) %out, i64 4
  store i32 %e, ptr addrspace(1) %p
  ret void
}
