; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Signatures are cloned before calls are retyped: cloning @f with its call to
; @g already returning the i32 placeholder would clone an ill-typed select.
; CHECK-DAG: [[F32:%.+]] = OpTypeFloat 32
; CHECK-DAG: [[PAIR:%.+]] = OpTypeStruct [[F32]] [[F32]]
; CHECK: [[R:%.+]] = OpFunctionCall [[PAIR]]
; CHECK: OpSelect [[PAIR]] {{%.+}} {{%.+}} [[R]]
define { float, float } @g(float %x) {
  %a = insertvalue { float, float } poison, float %x, 0
  %b = insertvalue { float, float } %a, float %x, 1
  ret { float, float } %b
}
define { float, float } @f(float %x, i1 %c) {
  %r = call { float, float } @g(float %x)
  %s = select i1 %c, { float, float } zeroinitializer, { float, float } %r
  ret { float, float } %s
}
