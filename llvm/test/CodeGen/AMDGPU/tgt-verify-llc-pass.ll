; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs -enable-new-pm -verify-each %s -o - 2>&1 | FileCheck %s

define amdgpu_cs void @void_shader() {
; CHECK: ModuleToFunctionPassAdaptor
  ret void
}
