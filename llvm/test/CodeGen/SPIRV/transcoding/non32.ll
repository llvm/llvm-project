; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: OpTypeInt 16
; CHECK: OpIAdd

define i16 @test_fn(i16 %arg0, i16 %arg1) {
entry:
  %0 = add i16 %arg0, %arg1
  ret i16 %0
}

declare spir_func i64 @_Z13get_global_idj(i32)
