; RUN: not llc -mtriple=amdgcn-amd-amdhsa < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: in function test_kernel{{.*}}: non-hsa intrinsic with hsa target
define amdgpu_kernel void @test_kernel(i32 addrspace(1)* %out) #1 {
  %implicit_buffer_ptr = call i8 addrspace(2)* @llvm.amdgcn.implicit.buffer.ptr()
  %header_ptr = bitcast i8 addrspace(2)* %implicit_buffer_ptr to i32 addrspace(2)*
  %value = load i32, i32 addrspace(2)* %header_ptr
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; ERROR: in function test_func{{.*}}: non-hsa intrinsic with hsa target
define void @test_func(i32 addrspace(1)* %out) #1 {
  %implicit_buffer_ptr = call i8 addrspace(2)* @llvm.amdgcn.implicit.buffer.ptr()
  %header_ptr = bitcast i8 addrspace(2)* %implicit_buffer_ptr to i32 addrspace(2)*
  %value = load i32, i32 addrspace(2)* %header_ptr
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

declare i8 addrspace(2)* @llvm.amdgcn.implicit.buffer.ptr() #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind  }
