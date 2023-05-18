; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

;; OpenCL global memory
define ptr addrspace(1) @getConstant1() {
  ret ptr addrspace(1) null
}

;; OpenCL constant memory
define ptr addrspace(2) @getConstant2() {
  ret ptr addrspace(2) null
}

;; OpenCL local memory
define ptr addrspace(3) @getConstant3() {
  ret ptr addrspace(3) null
}

; CHECK:     [[INT:%.+]] = OpTypeInt 8

; CHECK-DAG: [[PTR_AS1:%.+]] = OpTypePointer CrossWorkgroup [[INT]]
; CHECK-DAG: OpConstantNull [[PTR_AS1]]

; CHECK-DAG: [[PTR_AS2:%.+]] = OpTypePointer UniformConstant [[INT]]
; CHECK-DAG: OpConstantNull [[PTR_AS2]]

; CHECK-DAG: [[PTR_AS3:%.+]] = OpTypePointer Workgroup [[INT]]
; CHECK-DAG: OpConstantNull [[PTR_AS3]]
