; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: {{.*}}: this instruction is only supported with the GLSL extended instruction set.

define hidden spir_func noundef i32 @_Z9test_funcj(<2 x float> noundef %0) local_unnamed_addr #0 {
  %2 = tail call i32 @llvm.spv.packhalf2x16.i32(<2 x float> %0)
  ret i32 %2
}

