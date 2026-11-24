; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan %s -o /dev/null 2>&1 | FileCheck %s

; Test that G_PTRMASK is errors for logical SPIR-V.

; CHECK: G_PTRMASK is not supported with logical SPIR-V

define void @test_ptrmask_i32(ptr addrspace(1) %ptr, i32 %mask, ptr addrspace(1) %out) #1 {
entry:
  %masked_ptr = call ptr addrspace(1) @llvm.ptrmask.p1.i32(ptr addrspace(1) %ptr, i32 %mask)
  store ptr addrspace(1) %masked_ptr, ptr addrspace(1) %out, align 8
  ret void
}

declare ptr addrspace(1) @llvm.ptrmask.p1.i32(ptr addrspace(1), i32)

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
