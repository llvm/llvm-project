; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -print-after-all -o - 2>&1 | FileCheck %s

; CHECK: *** IR Dump After SPIRV emit intrinsics (emit-intrinsics) ***

define spir_kernel void @test(ptr addrspace(1) %srcimg) {
; CHECK: call void @llvm.spv.assign.type.p1(ptr addrspace(1) %srcimg, metadata target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) undef)
  %call1 = call spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(ptr addrspace(1) %srcimg)
; CHECK-NOT: call void @llvm.spv.assign.type.p1(ptr addrspace(1) %srcimg, metadata target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) undef)
  %call2 = call spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(ptr addrspace(1) %srcimg)
  ret void
; CHECK: }
}

declare spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(ptr addrspace(1))
