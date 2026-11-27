; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -stop-after=spirv-emit-intrinsics -o - | FileCheck %s

define spir_kernel void @test(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %srcimg) {
; CHECK-NOT: call void @llvm.spv.assign.type.p1(ptr addrspace(1) %srcimg, metadata target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) undef)
  %call = call spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %srcimg)
  ret void
; CHECK: }
}

declare spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0))
