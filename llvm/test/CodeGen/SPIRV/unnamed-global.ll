; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[TyInt:.*]] = OpTypeInt 8 0
; CHECK: %[[ConstInt:.*]] = OpConstant %[[TyInt]] 123
; CHECK: %[[TyPtr:.*]] = OpTypePointer {{[a-zA-Z]+}} %[[TyInt]]
; CHECK: %[[VarId:.*]] = OpVariable %[[TyPtr]] {{[a-zA-Z]+}} %[[ConstInt]]

@0 = addrspace(1) global i8 123

; Function Attrs: nounwind
define spir_kernel void @foo() #0 {
  ret void
}

attributes #0 = { nounwind }
