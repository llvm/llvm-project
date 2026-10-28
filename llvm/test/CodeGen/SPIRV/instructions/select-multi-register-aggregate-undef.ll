; A multi-register aggregate arm of a select (here from llvm.sincos) is rebuilt
; with insertvalue. The rebuilt chain must start from a typed placeholder: a
; bare poison became the uniqued i32 undef, which then merged with the i32 undef
; member of an unrelated constant and gave it the struct's type.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Long:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#Mixed:]] = OpTypeStruct %[[#Long]] %[[#Int]] %[[#Float]]
; CHECK-DAG: %[[#Pair:]] = OpTypeStruct %[[#Float]] %[[#Float]]
; CHECK-DAG: %[[#UndefInt:]] = OpUndef %[[#Int]]
; CHECK-DAG: %[[#UndefFloat:]] = OpUndef %[[#Float]]
; CHECK-DAG: %[[#UndefPair:]] = OpUndef %[[#Pair]]
; CHECK-DAG: %[[#Const:]] = OpConstantComposite %[[#Mixed]] %[[#]] %[[#UndefInt]] %[[#UndefFloat]]
; CHECK: %[[#Ins:]] = OpCompositeInsert %[[#Pair]] %[[#]] %[[#UndefPair]] 0
; CHECK: OpSelect %[[#Pair]] %[[#]] %[[#]] %[[#]]
; CHECK: OpCompositeInsert %[[#Mixed]] %[[#]] %[[#Const]] 1

define spir_kernel void @k(ptr addrspace(1) %out, float %a, i1 %c) {
  %sc = call { float, float } @llvm.sincos.f32(float %a)
  %sel = select i1 %c, { float, float } zeroinitializer, { float, float } %sc
  %s = extractvalue { float, float } %sel, 0
  store float %s, ptr addrspace(1) %out
  %agg = insertvalue { i64, i32, float } { i64 1, i32 undef, float undef }, i32 0, 1
  %f = extractvalue { i64, i32, float } %agg, 2
  %p = getelementptr inbounds i8, ptr addrspace(1) %out, i64 4
  store float %f, ptr addrspace(1) %p
  ret void
}
