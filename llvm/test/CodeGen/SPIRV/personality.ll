; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: __gxx_personality_v0
; CHECK: OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
; CHECK-NOT: OpFunction

define spir_func void @foo() addrspace(4) personality ptr addrspace(4) @__gxx_personality_v0 {
  ret void
}

declare i32 @__gxx_personality_v0(...) addrspace(4)
