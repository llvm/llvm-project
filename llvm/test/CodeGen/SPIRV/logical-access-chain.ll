; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG:      [[uint:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG:     [[uint2:%[0-9]+]] = OpTypeVector [[uint]] 2
; CHECK-DAG:    [[uint_1:%[0-9]+]] = OpConstant [[uint]] 1
; CHECK-DAG:  [[ptr_uint:%[0-9]+]] = OpTypePointer Function [[uint]]
; CHECK-DAG: [[ptr_uint2:%[0-9]+]] = OpTypePointer Function [[uint2]]

define void @main() #1 {
entry:
  %0 = alloca <2 x i32>, align 4
; CHECK: [[var:%[0-9]+]] = OpVariable [[ptr_uint2]] Function

  %1 = getelementptr <2 x i32>, ptr %0, i32 0, i32 1
; CHECK: {{%[0-9]+}} = OpAccessChain [[ptr_uint]] [[var]] [[uint_1]]

  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
