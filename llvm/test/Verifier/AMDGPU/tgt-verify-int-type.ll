; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-verify-tgt -o - < %s 2>&1 | FileCheck %s

define amdgpu_cs i65 @invalid_int() {
; CHECK: Int type is invalid.
  ret i65 0
}
