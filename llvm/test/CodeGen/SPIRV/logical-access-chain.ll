; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG:      [[uint:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG:     [[uint2:%[0-9]+]] = OpTypeVector [[uint]] 2
; CHECK-DAG:    [[uint_1:%[0-9]+]] = OpConstant [[uint]] 1
; CHECK-DAG:    [[uint_2:%[0-9]+]] = OpConstant [[uint]] 2
; CHECK-DAG:  [[ptr_uint:%[0-9]+]] = OpTypePointer Private [[uint]]
; CHECK-DAG: [[ptr_uint2:%[0-9]+]] = OpTypePointer Private [[uint2]]

; CHECK-DAG: [[var:%[0-9]+]] = OpVariable [[ptr_uint2]] Private

define void @main() #1 {
entry:
  %0 = alloca <2 x i32>, align 4
; CHECK:     OpLabel
; CHECK-NOT: OpVariable

  %1 = getelementptr <2 x i32>, ptr %0, i32 0, i32 1
; CHECK: [[tmp:%[0-9]+]] = OpAccessChain [[ptr_uint]] [[var]] [[uint_1]]

  store i32 2, ptr %1
; CHECK: OpStore [[tmp]] [[uint_2]] Aligned 4

  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
