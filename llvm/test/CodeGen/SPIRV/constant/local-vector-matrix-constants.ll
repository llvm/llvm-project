; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

;; TODO: Add test for matrix. But how are they represented in LLVM IR?

define <4 x i8> @getVectorConstant() {
  ret <4 x i8> <i8 1, i8 1, i8 1, i8 1>
}

; CHECK-DAG: [[I8:%.+]] = OpTypeInt 8
; CHECK-DAG: [[VECTOR:%.+]] = OpTypeVector [[I8]]
; CHECK-DAG: [[CST_I8:%.+]] = OpConstant [[I8]] 1
; CHECK-DAG: [[CST_VECTOR:%.+]] = OpConstantComposite [[VECTOR]] [[CST_I8]] [[CST_I8]] [[CST_I8]] [[CST_I8]]
