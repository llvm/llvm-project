; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG:  %[[#FLOAT32:]] = OpTypeFloat 32
; CHECK-DAG:  %[[#PTR1:]] = OpTypePointer Function %[[#FLOAT32]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR1]]

define spir_kernel void @test1(ptr %arg) !kernel_arg_type !1 {
  %a = getelementptr inbounds float, ptr %arg, i64 1
  ret void
}

!1 = !{!"float*"}

; CHECK-DAG:  %[[#CHAR:]] = OpTypeInt 8 0
; CHECK-DAG:  %[[#PTR2:]] = OpTypePointer Function %[[#CHAR]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR2]]

define spir_kernel void @test2(ptr %arg) !kernel_arg_type !2 {
  %a = getelementptr inbounds i8, ptr %arg, i64 1
  ret void
}

!2 = !{!"char*"}

; CHECK-DAG:  %[[#SHORT:]] = OpTypeInt 16 0
; CHECK-DAG:  %[[#PTR3:]] = OpTypePointer Function %[[#SHORT]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR3]]

define spir_kernel void @test3(ptr %arg) !kernel_arg_type !3 {
  %a = getelementptr inbounds i16, ptr %arg, i64 1
  ret void
}

!3 = !{!"short*"}

; CHECK-DAG:  %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG:  %[[#PTR4:]] = OpTypePointer Function %[[#INT]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR4]]

define spir_kernel void @test4(ptr %arg) !kernel_arg_type !4 {
  %a = getelementptr inbounds i32, ptr %arg, i64 1
  ret void
}

!4 = !{!"int*"}

; CHECK-DAG:  %[[#LONG:]] = OpTypeInt 64 0
; CHECK-DAG:  %[[#PTR5:]] = OpTypePointer Function %[[#LONG]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR5]]

define spir_kernel void @test5(ptr %arg) !kernel_arg_type !5 {
  %a = getelementptr inbounds i64, ptr %arg, i64 1
  ret void
}

!5 = !{!"long*"}

; CHECK-DAG:  %[[#DOUBLE:]] = OpTypeFloat 64
; CHECK-DAG:  %[[#PTR6:]] = OpTypePointer Function %[[#DOUBLE]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR6]]

define spir_kernel void @test6(ptr %arg) !kernel_arg_type !6 {
  %a = getelementptr inbounds double, ptr %arg, i64 1
  ret void
}

!6 = !{!"double*"}

; CHECK-DAG:  %[[#HALF:]] = OpTypeFloat 16
; CHECK-DAG:  %[[#PTR7:]] = OpTypePointer Function %[[#HALF]]
; CHECK-DAG:  %[[#ARG:]] = OpFunctionParameter %[[#PTR7]]

define spir_kernel void @test7(ptr %arg) !kernel_arg_type !7 {
  %a = getelementptr inbounds half, ptr %arg, i64 1
  ret void
}

!7 = !{!"half*"}
