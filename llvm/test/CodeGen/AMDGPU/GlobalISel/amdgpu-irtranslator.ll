; RUN: llc -march=amdgcn -mcpu=fiji -O0 -stop-after=irtranslator -global-isel %s -o - 2>&1 | FileCheck %s
; REQUIRES: global-isel
; This file checks that the translation from llvm IR to generic MachineInstr
; is correct.

; Tests for add.
; CHECK: name: addi32
; CHECK: G_ADD i32
define i32 @addi32(i32 %arg1, i32 %arg2) {
  %res = add i32 %arg1, %arg2
  ret i32 %res
}
