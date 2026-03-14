; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Type mismatch {{.*}}

define spir_kernel void @test(ptr addrspace(1) %srcimg) {
  %call1 = call spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(ptr addrspace(1) %srcimg)
  %call2 = call spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_rw(ptr addrspace(1) %srcimg)
  ret void
}

declare spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(ptr addrspace(1))
declare spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_rw(ptr addrspace(1))
