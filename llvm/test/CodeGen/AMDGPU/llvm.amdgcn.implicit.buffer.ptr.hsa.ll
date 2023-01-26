; RUN: not llc -mtriple=amdgcn-amd-amdhsa < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: in function test_kernel{{.*}}: non-hsa intrinsic with hsa target
define amdgpu_kernel void @test_kernel(ptr addrspace(1) %out) #1 {
  %implicit_buffer_ptr = call ptr addrspace(4) @llvm.amdgcn.implicit.buffer.ptr()
  %value = load i32, ptr addrspace(4) %implicit_buffer_ptr
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; ERROR: in function test_func{{.*}}: non-hsa intrinsic with hsa target
define void @test_func(ptr addrspace(1) %out) #1 {
  %implicit_buffer_ptr = call ptr addrspace(4) @llvm.amdgcn.implicit.buffer.ptr()
  %value = load i32, ptr addrspace(4) %implicit_buffer_ptr
  store i32 %value, ptr addrspace(1) %out
  ret void
}

declare ptr addrspace(4) @llvm.amdgcn.implicit.buffer.ptr() #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind  }
