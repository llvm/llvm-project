; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown < %s 2>&1 | FileCheck -check-prefix=CHECK-ERR %s

; For compute, nothing is generated, but compilation doesn't crash.
; CHECK: OpName %[[#FOO:]] "foo"
; CHECK: OpName %[[#RTM:]] "reg2mem alloca point"
; CHECK: %[[#INT:]] = OpTypeInt 32 0
; CHECK: %[[#RTM]] = OpConstant %[[#INT]] 0
; CHECK: %[[#FOO]] = OpFunction
; CHECK-NEXT: = OpLabel
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd


; For non-compute, error.
; CHECK-ERR: LLVM ERROR: Runtime arrays are not allowed in non-shader SPIR-V modules

define spir_func void @foo() {
entry:
  %i = alloca [0 x [ 0 x i32]], align 4
  ret void
}
