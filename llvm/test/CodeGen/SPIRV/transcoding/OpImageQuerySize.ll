; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; Check conversion of get_image_width, get_image_height, get_image_depth,
;; get_image_array_size, and get_image_dim OCL built-ins.
;; In general the SPRI-V reader converts OpImageQuerySize into get_image_dim
;; and subsequent extract or shufflevector instructions. Unfortunately there is
;; no get_image_dim for 1D images and get_image_dim cannot replace get_image_array_size

; CHECK-SPIRV: %[[#ArrayTypeID:]] = OpTypeImage %[[#]] 1D 0 1 0 0 Unknown ReadOnly

; CHECK-SPIRV:     %[[#ArrayVarID:]] = OpFunctionParameter %[[#ArrayTypeID]]
; CHECK-SPIRV:     %[[#]] = OpImageQuerySizeLod %[[#]] %[[#ArrayVarID]]
; CHECK-SPIRV-NOT: %[[#]] = OpExtInst %[[#]] %[[#]] get_image_array_size

define spir_kernel void @test_image1d(i32 addrspace(1)* nocapture %sizes, target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %img, target("spirv.Image", void, 5, 0, 0, 0, 0, 0, 0) %buffer, target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0) %array) {
  %1 = tail call spir_func i32 @_Z15get_image_width14ocl_image1d_ro(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %img)
  %2 = tail call spir_func i32 @_Z15get_image_width21ocl_image1d_buffer_ro(target("spirv.Image", void, 5, 0, 0, 0, 0, 0, 0) %buffer)
  %3 = tail call spir_func i32 @_Z15get_image_width20ocl_image1d_array_ro(target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0) %array)
  %4 = tail call spir_func i64 @_Z20get_image_array_size20ocl_image1d_array_ro(target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0) %array)
  %5 = trunc i64 %4 to i32
  %6 = add nsw i32 %2, %1
  %7 = add nsw i32 %6, %3
  %8 = add nsw i32 %7, %5
  store i32 %8, i32 addrspace(1)* %sizes, align 4
  ret void
}

declare spir_func i32 @_Z15get_image_width14ocl_image1d_ro(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0))

declare spir_func i32 @_Z15get_image_width21ocl_image1d_buffer_ro(target("spirv.Image", void, 5, 0, 0, 0, 0, 0, 0))

declare spir_func i32 @_Z15get_image_width20ocl_image1d_array_ro(target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0))

declare spir_func i64 @_Z20get_image_array_size20ocl_image1d_array_ro(target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0))

define spir_kernel void @test_image2d(i32 addrspace(1)* nocapture %sizes, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img, target("spirv.Image", void, 1, 1, 0, 0, 0, 0, 0) %img_depth, target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0) %array, target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0) %array_depth) {
  %1 = tail call spir_func i32 @_Z15get_image_width14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img)
  %2 = tail call spir_func i32 @_Z16get_image_height14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img)
  %3 = tail call spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img)
  %4 = tail call spir_func i32 @_Z15get_image_width20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0) %array)
  %5 = tail call spir_func i32 @_Z16get_image_height20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0) %array)
  %6 = tail call spir_func i64 @_Z20get_image_array_size20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0) %array)
  %7 = trunc i64 %6 to i32
  %8 = tail call spir_func <2 x i32> @_Z13get_image_dim20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0) %array)
  %9 = add nsw i32 %2, %1
  %10 = extractelement <2 x i32> %3, i32 0
  %11 = add nsw i32 %9, %10
  %12 = extractelement <2 x i32> %3, i32 1
  %13 = add nsw i32 %11, %12
  %14 = add nsw i32 %13, %4
  %15 = add nsw i32 %14, %5
  %16 = add nsw i32 %15, %7
  %17 = extractelement <2 x i32> %8, i32 0
  %18 = add nsw i32 %16, %17
  %19 = extractelement <2 x i32> %8, i32 1
  %20 = add nsw i32 %18, %19
  store i32 %20, i32 addrspace(1)* %sizes, align 4
  ret void
}

declare spir_func i32 @_Z15get_image_width14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0))

declare spir_func i32 @_Z16get_image_height14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0))

declare spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0))

declare spir_func i32 @_Z15get_image_width20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0))

declare spir_func i32 @_Z16get_image_height20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0))

declare spir_func i64 @_Z20get_image_array_size20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0))

declare spir_func <2 x i32> @_Z13get_image_dim20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0))

define spir_kernel void @test_image3d(i32 addrspace(1)* nocapture %sizes, target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %img) {
  %1 = tail call spir_func i32 @_Z15get_image_width14ocl_image3d_ro(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %img)
  %2 = tail call spir_func i32 @_Z16get_image_height14ocl_image3d_ro(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %img)
  %3 = tail call spir_func i32 @_Z15get_image_depth14ocl_image3d_ro(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %img)
  %4 = tail call spir_func <4 x i32> @_Z13get_image_dim14ocl_image3d_ro(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %img)
  %5 = add nsw i32 %2, %1
  %6 = add nsw i32 %5, %3
  %7 = extractelement <4 x i32> %4, i32 0
  %8 = add nsw i32 %6, %7
  %9 = extractelement <4 x i32> %4, i32 1
  %10 = add nsw i32 %8, %9
  %11 = extractelement <4 x i32> %4, i32 2
  %12 = add nsw i32 %10, %11
  %13 = extractelement <4 x i32> %4, i32 3
  %14 = add nsw i32 %12, %13
  store i32 %14, i32 addrspace(1)* %sizes, align 4
  ret void
}

declare spir_func i32 @_Z15get_image_width14ocl_image3d_ro(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0))

declare spir_func i32 @_Z16get_image_height14ocl_image3d_ro(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0))

declare spir_func i32 @_Z15get_image_depth14ocl_image3d_ro(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0))

declare spir_func <4 x i32> @_Z13get_image_dim14ocl_image3d_ro(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0))

define spir_kernel void @test_image2d_array_depth_t(i32 addrspace(1)* nocapture %sizes, target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0) %array) {
  %1 = tail call spir_func i32 @_Z15get_image_width26ocl_image2d_array_depth_ro(target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0) %array)
  %2 = tail call spir_func i32 @_Z16get_image_height26ocl_image2d_array_depth_ro(target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0) %array)
  %3 = tail call spir_func i64 @_Z20get_image_array_size26ocl_image2d_array_depth_ro(target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0) %array)
  %4 = trunc i64 %3 to i32
  %5 = add nsw i32 %2, %1
  %6 = add nsw i32 %5, %4
  store i32 %5, i32 addrspace(1)* %sizes, align 4
  ret void
}

declare spir_func i32 @_Z15get_image_width26ocl_image2d_array_depth_ro(target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0))

declare spir_func i32 @_Z16get_image_height26ocl_image2d_array_depth_ro(target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0))

declare spir_func i64 @_Z20get_image_array_size26ocl_image2d_array_depth_ro(target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0))
