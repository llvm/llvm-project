; RUN: llc -O0 -mtriple=spirv32-unknown-unknown < %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown < %s -filetype=obj | not spirv-val 2>&1 | FileCheck %s --check-prefix=VALIDATOR %}
;
; _Z3miniii is not a valid OpenCL intrinsic, do not treat it like one.
;
; VALIDATOR: Invalid instruction OpExtInst starting at word {{[0-9]+}}: expected no more operands after 7 words, but stated word count is 8

define spir_kernel void @ill_1() {
; CHECK-LABEL:   OpFunction %{{[0-9]+}} None %{{[0-9]+}} ; -- Begin function ill_1
; CHECK-NEXT:    OpLabel
; This is wrong, we should generate a regular call
; CHECK-NEXT:    %{{[0-9]+}} = OpExtInst %{{[0-9]+}} %{{[0-9]+}} s_min %{{[0-9]+}} %{{[0-9]+}} %{{[0-9]+}}
; CHECK-NEXT:    OpReturn
; CHECK-NEXT:    OpFunctionEnd
; CHECK-NEXT:    ; -- End function
entry:
  tail call spir_func void @_Z3miniii(i32 1, i32 2, i32 3)
  ret void
}

declare spir_func i32 @_Z3miniii(i32, i32, i32)
