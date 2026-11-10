; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=ir_name --implicit-check-not=ignored
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#A:]] "a"
; CHECK-DAG: OpName %[[#B:]] "b"
; CHECK-DAG: OpName %[[#X:]] "x"
; CHECK-DAG: OpName %[[#M:]] "m"
; CHECK-DAG: OpName %[[#K:]] "kernel"
; CHECK-DAG: OpName %[[#F:]] "func"
; CHECK-DAG: OpName %[[#MK:]] "malformed"

; CHECK: %[[#K]] = OpFunction
; CHECK-NEXT: %[[#A]] = OpFunctionParameter
; CHECK-NEXT: %[[#B]] = OpFunctionParameter
; CHECK-NEXT: OpFunctionParameter
define spir_kernel void @kernel(i32, i32 %ir_name, i32 %ir_name2) !kernel_arg_name !0 {
  ret void
}

; CHECK: %[[#F]] = OpFunction
; CHECK-NEXT: %[[#X]] = OpFunctionParameter
define spir_func void @func(i32 %x) !kernel_arg_name !1 {
  ret void
}

; CHECK: %[[#MK]] = OpFunction
; CHECK-NEXT: %[[#M]] = OpFunctionParameter
define spir_kernel void @malformed(i32 %m) !kernel_arg_name !2 {
  ret void
}

!0 = !{!"a", !"b", !""}
!1 = !{!"ignored"}
!2 = !{i32 0}
