; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

%aggregate = type { i8, i32 }

define %aggregate @getConstant() {
  ret %aggregate { i8 1, i32 2 }
}

; CHECK:     OpName [[GET:%.+]] "getConstant"

; CHECK-DAG: [[I8:%.+]] = OpTypeInt 8
; CHECK-DAG: [[I32:%.+]] = OpTypeInt 32
; CHECK-DAG: [[AGGREGATE:%.+]] = OpTypeStruct [[I8]] [[I32]]
; CHECK-DAG: [[CST_I8:%.+]] = OpConstant [[I8]] 1
; CHECK-DAG: [[CST_I32:%.+]] = OpConstant [[I32]] 2
; CHECK-DAG: [[CST_AGGREGATE:%.+]] = OpConstantComposite [[AGGREGATE]] [[CST_I8]] [[CST_I32]]

; CHECK:     [[GET]] = OpFunction [[AGGREGATE]]
; CHECK:     OpReturnValue [[CST_AGGREGATE]]
; CHECK:     OpFunctionEnd
