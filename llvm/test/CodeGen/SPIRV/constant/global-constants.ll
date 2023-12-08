; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

@global   = addrspace(1) constant i32 1 ; OpenCL global memory
@constant = addrspace(2) constant i32 2 ; OpenCL constant memory
@local    = addrspace(3) constant i32 3 ; OpenCL local memory

define i32 @getGlobal1() {
  %g = load i32, i32 addrspace(1)* @global
  ret i32 %g
}

define i32 @getGlobal2() {
  %g = load i32, i32 addrspace(2)* @constant
  ret i32 %g
}

define i32 @getGlobal3() {
  %g = load i32, i32 addrspace(3)* @local
  ret i32 %g
}

; CHECK:     [[INT:%.+]] = OpTypeInt 32

; CHECK-DAG: [[PTR_TO_INT_AS1:%.+]] = OpTypePointer CrossWorkgroup [[INT]]
; CHECK-DAG: [[PTR_TO_INT_AS2:%.+]] = OpTypePointer UniformConstant [[INT]]
; CHECK-DAG: [[PTR_TO_INT_AS3:%.+]] = OpTypePointer Workgroup [[INT]]

; CHECK-DAG: [[CST_AS1:%.+]] = OpConstant [[INT]] 1
; CHECK-DAG: [[CST_AS2:%.+]] = OpConstant [[INT]] 2
; CHECK-DAG: [[CST_AS3:%.+]] = OpConstant [[INT]] 3

; CHECK-DAG: [[GV1:%.+]] = OpVariable [[PTR_TO_INT_AS1]] CrossWorkgroup [[CST_AS1]]
; CHECK-DAG: [[GV2:%.+]] = OpVariable [[PTR_TO_INT_AS2]] UniformConstant [[CST_AS2]]
; CHECK-DAG: [[GV3:%.+]] = OpVariable [[PTR_TO_INT_AS3]] Workgroup [[CST_AS3]]

; CHECK:     OpLoad [[INT]] [[GV1]]
; CHECK:     OpLoad [[INT]] [[GV2]]
; CHECK:     OpLoad [[INT]] [[GV3]]
