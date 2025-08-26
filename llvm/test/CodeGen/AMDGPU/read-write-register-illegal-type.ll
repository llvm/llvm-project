; RUN: not llc -mtriple=amdgcn -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error: <unknown>:0:0: cannot use llvm.read_register with illegal type
define amdgpu_kernel void @test_read_register_i9(ptr addrspace(1) %out) nounwind {
  %reg = call i9 @llvm.read_register.i9(metadata !0)
  store i9 %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: cannot use llvm.write_register with illegal type
define amdgpu_kernel void @test_write_register_i9(ptr addrspace(1) %out) nounwind {
 call void @llvm.write_register.i9(metadata !0, i9 42)
 ret void
}

; CHECK: error: <unknown>:0:0: cannot use llvm.read_register with illegal type
define amdgpu_kernel void @test_read_register_i128(ptr addrspace(1) %out) nounwind {
  %reg = call i128 @llvm.read_register.i128(metadata !0)
  store i128 %reg, ptr addrspace(1) %out
  ret void
}

; CHECK: error: <unknown>:0:0: cannot use llvm.write_register with illegal type
define amdgpu_kernel void @test_write_register_i128(ptr addrspace(1) %out) nounwind {
 call void @llvm.write_register.i128(metadata !0, i128 42)
 ret void
}

!0 = !{!"m0"}
