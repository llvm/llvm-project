; XFAIL: target={{.*}}-aix{{.*}}

; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds %s -o %t.ll
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds %t.ll -o %t.second.ll
; RUN: diff -ub %t.ll %t.second.ll -I ".*ModuleID.*"

; Check AMDGPULowerModuleLDS can run more than once on the same module, and that
; the second run is a no-op.

@lds = internal unnamed_addr addrspace(3) global i32 undef, align 4

define amdgpu_kernel void @test() {
entry:
  store i32 1, ptr addrspace(3) @lds
  ret void
}
