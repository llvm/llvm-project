; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck %s

; Check that we do not assume any default stack size for PAL code object
; indirect calls. The driver knows the max recursion depth, so it can compute
; a more accurate value.

; CHECK: ScratchSize: 0
; CHECK: scratch_memory_size: 0
define amdgpu_vs void @test() {
.entry:
  %0 = call i64 @llvm.amdgcn.s.getpc()
  %1 = inttoptr i64 %0 to ptr
  call amdgpu_gfx void %1()
  ret void
}

declare i64 @llvm.amdgcn.s.getpc()
