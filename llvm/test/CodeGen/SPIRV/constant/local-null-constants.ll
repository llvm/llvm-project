; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

;; OpenCL global memory
define i32 addrspace(1)* @getConstant1() {
  ret i32 addrspace(1)* null
}

;; OpenCL constant memory
define i32 addrspace(2)* @getConstant2() {
  ret i32 addrspace(2)* null
}

;; OpenCL local memory
define i32 addrspace(3)* @getConstant3() {
  ret i32 addrspace(3)* null
}

; CHECK:     [[INT:%.+]] = OpTypeInt 32

; CHECK-DAG: [[PTR_AS1:%.+]] = OpTypePointer CrossWorkgroup [[INT]]
; CHECK-DAG: OpConstantNull [[PTR_AS1]]

; CHECK-DAG: [[PTR_AS2:%.+]] = OpTypePointer UniformConstant [[INT]]
; CHECK-DAG: OpConstantNull [[PTR_AS2]]

; CHECK-DAG: [[PTR_AS3:%.+]] = OpTypePointer Workgroup [[INT]]
; CHECK-DAG: OpConstantNull [[PTR_AS3]]
