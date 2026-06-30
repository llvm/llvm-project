; RUN: not llc -mtriple=amdgcn -mcpu=gfx1200 -o - < %s 2>&1 | FileCheck %s

define amdgpu_cs_chain void @test_alloca() {
; CHECK: Chain calling convention is invalid on this target
  ret void
}
