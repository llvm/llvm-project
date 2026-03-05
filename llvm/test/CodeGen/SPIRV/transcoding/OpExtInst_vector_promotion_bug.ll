; RUN: llc -O0 -mtriple=spirv32-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown < %s -filetype=obj | spirv-val %}

define spir_kernel void @ill_1() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function ill_1
; CHECK-NEXT:    OpLabel
; CHECK-NEXT:    %{{[0-9]+}} = OpFunctionCall %{{[0-9]+}} %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  tail call spir_func void @_Z3miniii(i32 1, i32 2, i32 3)
  ret void
}

declare spir_func i32 @_Z3miniii(i32, i32, i32)
