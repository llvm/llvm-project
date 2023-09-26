; RUN: llc -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG:      [[uint:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG:      [[bool:%[0-9]+]] = OpTypeBool
; CHECK-DAG:  [[ptr_uint:%[0-9]+]] = OpTypePointer Function [[uint]]
; CHECK-DAG:    [[uint_0:%[0-9]+]] = OpConstant [[uint]] 0
; CHECK-DAG:    [[uint_10:%[0-9]+]] = OpConstant [[uint]] 10
; CHECK-DAG:    [[uint_20:%[0-9]+]] = OpConstant [[uint]] 20

define void @main() #1 {
  %input = alloca i32, align 4
  %output = alloca i32, align 4
; CHECK: [[input:%[0-9]+]] = OpVariable [[ptr_uint]] Function
; CHECK: [[output:%[0-9]+]] = OpVariable [[ptr_uint]] Function

  %1 = load i32, i32* %input, align 4
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %true, label %false
; CHECK:  [[tmp:%[0-9]+]] = OpLoad [[uint]] [[input]]
; CHECK: [[cond:%[0-9]+]] = OpINotEqual [[bool]] [[tmp]] [[uint_0]]
; CHECK:                    OpBranchConditional [[cond]] [[true:%[0-9]+]] [[false:%[0-9]+]]

true:
  store i32 10, i32* %output, align 4
  br label %merge
; CHECK: [[true]] = OpLabel
; CHECK:            OpStore [[output]] [[uint_10]]
; CHECK:            OpBranch [[merge:%[0-9]+]]

false:
  store i32 20, i32* %output, align 4
  br label %merge
; CHECK: [[false]] = OpLabel
; CHECK:            OpStore [[output]] [[uint_20]]
; CHECK:            OpBranch [[merge]]

merge:
; CHECK: [[merge]] = OpLabel
; CHECK:             OpReturn
  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" convergent }
