; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; Nothing is generated, but compilation doesn't crash.
; CHECK: OpName %[[#FOO:]] "foo"
; CHECK: OpName %[[#RTM:]] "reg2mem alloca point"
; CHECK: %[[#INT:]] = OpTypeInt 32 0
; CHECK: %[[#RTM]] = OpConstant %[[#INT]] 0
; CHECK: %[[#FOO]] = OpFunction
; CHECK-NEXT: = OpLabel
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd

define spir_func void @foo() {
entry:
  %i = alloca [0 x i32], align 4
  ret void
}
