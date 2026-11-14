; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=ir_name --implicit-check-not=ignored
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Kernel parameter OpNames come from kernel_arg_name metadata, else IR names.

; CHECK: OpName %[[#A:]] "a"
; CHECK: OpName %[[#B:]] "b"
; CHECK: OpName %[[#K:]] "kernel"
; CHECK: OpName %[[#X:]] "x"
; CHECK: OpName %[[#F:]] "func"
; CHECK: OpName %[[#M:]] "m"
; CHECK: OpName %[[#MK:]] "malformed"
; CHECK: OpName %[[#Y:]] "y"
; CHECK: OpName %[[#N:]] "n"
; CHECK: OpName %[[#KM:]] "kernels_md"

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

; CHECK: %[[#KM]] = OpFunction
; CHECK-NEXT: %[[#Y]] = OpFunctionParameter
; CHECK-NEXT: %[[#N]] = OpFunctionParameter
define spir_kernel void @kernels_md(i32 %ir_name, i32 %n) {
  ret void
}

!opencl.kernels = !{!3}

!0 = !{!"a", !"b", !""}
!1 = !{!"ignored"}
!2 = !{i32 0}
!3 = !{ptr @kernels_md, !4}
!4 = !{!"kernel_arg_name", !"y", i32 0}
