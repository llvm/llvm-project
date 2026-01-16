; RUN: llc -O0 -mtriple=spirv64-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown -filetype=obj < %s  | spirv-val %}

@bar_alias = alias void (), ptr addrspace(4) @bar

define spir_func void @bar() addrspace(4) {
; CHECK:         %6 = OpFunction %4 None %5 ; -- Begin function bar
; CHECK-NEXT:    %2 = OpLabel
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
entry:
  ret void
}

define spir_kernel void @kernel() addrspace(4) {
entry:
; CHECK: %7 = OpFunction %4 None %5              ; -- Begin function kernel
; CHECK-NEXT: %3 = OpLabel
; CHECK-NEXT: %8 = OpFunctionCall %4 %6
  call spir_func addrspace(4) void @bar_alias()
  ret void
}
