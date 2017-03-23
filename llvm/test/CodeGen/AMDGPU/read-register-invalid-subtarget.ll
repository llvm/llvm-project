; RUN: not llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s 2>&1 | FileCheck %s

; CHECK: invalid register "flat_scratch_lo" for subtarget.

declare i32 @llvm.read_register.i32(metadata) #0

define amdgpu_kernel void @test_invalid_read_flat_scratch_lo(i32 addrspace(1)* %out) nounwind {
  store volatile i32 0, i32 addrspace(3)* undef
  %m0 = call i32 @llvm.read_register.i32(metadata !0)
  store i32 %m0, i32 addrspace(1)* %out
  ret void
}

!0 = !{!"flat_scratch_lo"}
