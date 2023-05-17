; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName [[VALUE:%.+]] "value"
; CHECK-DAG: OpName [[IDENTITY:%.+]] "identity"
; CHECK-DAG: OpName [[FOO:%.+]] "foo"

; CHECK:     [[INT:%.+]] = OpTypeInt 32
; CHECK-DAG: [[CST:%.+]] = OpConstant [[INT]] 42

define i32 @identity(i32 %value) {
  ret i32 %value
}

define i32 @foo() {
  %x = call i32 @identity(i32 42)
  ret i32 %x
}

; CHECK: [[FOO]] = OpFunction [[INT]]
; CHECK: [[X:%.+]] = OpFunctionCall [[INT]] [[IDENTITY]] [[CST]]
; CHECK: OpReturnValue [[X]]
; CHECK: OpFunctionEnd
