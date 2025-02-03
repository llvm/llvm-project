; RUN: not not llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs -enable-new-pm -verify-each -o - < %s 2>&1 | FileCheck %s

define amdgpu_cs i32 @nonvoid_shader() {
; CHECK: LLVM ERROR
  ret i32 0
}
