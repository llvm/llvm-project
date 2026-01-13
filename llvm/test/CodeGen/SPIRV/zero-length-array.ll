; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown < %s 2>&1 | FileCheck -check-prefix=CHECK-ERR %s

; For compute, nothing is generated, but compilation doesn't crash.
; CHECK: OpName %[[#FOO:]] "foo"
; CHECK: %[[#FOO]] = OpFunction
; CHECK-NEXT: = OpLabel
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd


; For non-compute, error.
; CHECK-ERR: LLVM ERROR: Runtime arrays are not allowed in non-shader SPIR-V modules

%struct.with_zero = type { i32, [0 x i32], i64 }

define spir_func void @foo() {
entry:
  %i = alloca %struct.with_zero, align 64
  ret void
}
