; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

;; Types:
; CHECK-DAG:  %[[#INT:]] = OpTypeInt 32
; CHECK-DAG:  %[[#VOID:]] = OpTypeVoid
; CHECK-DAG:  %[[#IMAGE1D_T:]] = OpTypeImage %[[#VOID]] 1D 0 0 0 0 Unknown ReadOnly
; CHECK-DAG:  %[[#IMAGE2D_T:]] = OpTypeImage %[[#VOID]] 2D 0 0 0 0 Unknown ReadOnly
; CHECK-DAG:  %[[#IMAGE3D_T:]] = OpTypeImage %[[#VOID]] 3D 0 0 0 0 Unknown ReadOnly
; CHECK-DAG:  %[[#IMAGE1D_ARRAY_T:]] = OpTypeImage %[[#VOID]] 1D 0 1 0 0 Unknown ReadOnly
; CHECK-DAG:  %[[#IMAGE2D_ARRAY_T:]] = OpTypeImage %[[#VOID]] 2D 0 1 0 0 Unknown ReadOnly
; CHECK-DAG:  %[[#IMAGE2D_DEPTH_T:]] = OpTypeImage %[[#VOID]] 2D 1 0 0 0 Unknown ReadOnly
; CHECK-DAG:  %[[#IMAGE2D_ARRAY_DEPTH_T:]] = OpTypeImage %[[#VOID]] 2D 1 1 0 0 Unknown ReadOnly
;; Instructions:
; CHECK:      %[[#IMAGE1D:]] = OpLoad %[[#IMAGE1D_T]]
; CHECK-NEXT: %[[#]] = OpImageQueryLevels %[[#INT]] %[[#IMAGE1D]]
; CHECK:      %[[#IMAGE2D:]] = OpLoad %[[#IMAGE2D_T]]
; CHECK-NEXT: %[[#]] = OpImageQueryLevels %[[#INT]] %[[#IMAGE2D]]
; CHECK:      %[[#IMAGE3D:]] = OpLoad %[[#IMAGE3D_T]]
; CHECK-NEXT: %[[#]] = OpImageQueryLevels %[[#INT]] %[[#IMAGE3D]]
; CHECK:      %[[#IMAGE1D_ARRAY:]] = OpLoad %[[#IMAGE1D_ARRAY_T]]
; CHECK-NEXT: %[[#]] = OpImageQueryLevels %[[#INT]] %[[#IMAGE1D_ARRAY]]
; CHECK:      %[[#IMAGE2D_ARRAY:]] = OpLoad %[[#IMAGE2D_ARRAY_T]]
; CHECK-NEXT: %[[#]] = OpImageQueryLevels %[[#INT]] %[[#IMAGE2D_ARRAY]]
; CHECK:      %[[#IMAGE2D_DEPTH:]] = OpLoad %[[#IMAGE2D_DEPTH_T]]
; CHECK-NEXT: %[[#]] = OpImageQueryLevels %[[#INT]] %[[#IMAGE2D_DEPTH]]
; CHECK:      %[[#IMAGE2D_ARRAY_DEPTH:]] = OpLoad %[[#IMAGE2D_ARRAY_DEPTH_T]]
; CHECK-NEXT: %[[#]] = OpImageQueryLevels %[[#INT]] %[[#IMAGE2D_ARRAY_DEPTH]]

%opencl.image1d_ro_t = type opaque
%opencl.image2d_ro_t = type opaque
%opencl.image3d_ro_t = type opaque
%opencl.image1d_array_ro_t = type opaque
%opencl.image2d_array_ro_t = type opaque
%opencl.image2d_depth_ro_t = type opaque
%opencl.image2d_array_depth_ro_t = type opaque

define spir_func void @testimage1d(%opencl.image1d_ro_t addrspace(1)* %img1, %opencl.image2d_ro_t addrspace(1)* %img2, %opencl.image3d_ro_t addrspace(1)* %img3, %opencl.image1d_array_ro_t addrspace(1)* %img4, %opencl.image2d_array_ro_t addrspace(1)* %img5, %opencl.image2d_depth_ro_t addrspace(1)* %img6, %opencl.image2d_array_depth_ro_t addrspace(1)* %img7) {
entry:
  %img1.addr = alloca %opencl.image1d_ro_t addrspace(1)*, align 4
  %img2.addr = alloca %opencl.image2d_ro_t addrspace(1)*, align 4
  %img3.addr = alloca %opencl.image3d_ro_t addrspace(1)*, align 4
  %img4.addr = alloca %opencl.image1d_array_ro_t addrspace(1)*, align 4
  %img5.addr = alloca %opencl.image2d_array_ro_t addrspace(1)*, align 4
  %img6.addr = alloca %opencl.image2d_depth_ro_t addrspace(1)*, align 4
  %img7.addr = alloca %opencl.image2d_array_depth_ro_t addrspace(1)*, align 4
  store %opencl.image1d_ro_t addrspace(1)* %img1, %opencl.image1d_ro_t addrspace(1)** %img1.addr, align 4
  store %opencl.image2d_ro_t addrspace(1)* %img2, %opencl.image2d_ro_t addrspace(1)** %img2.addr, align 4
  store %opencl.image3d_ro_t addrspace(1)* %img3, %opencl.image3d_ro_t addrspace(1)** %img3.addr, align 4
  store %opencl.image1d_array_ro_t addrspace(1)* %img4, %opencl.image1d_array_ro_t addrspace(1)** %img4.addr, align 4
  store %opencl.image2d_array_ro_t addrspace(1)* %img5, %opencl.image2d_array_ro_t addrspace(1)** %img5.addr, align 4
  store %opencl.image2d_depth_ro_t addrspace(1)* %img6, %opencl.image2d_depth_ro_t addrspace(1)** %img6.addr, align 4
  store %opencl.image2d_array_depth_ro_t addrspace(1)* %img7, %opencl.image2d_array_depth_ro_t addrspace(1)** %img7.addr, align 4
  %0 = load %opencl.image1d_ro_t addrspace(1)*, %opencl.image1d_ro_t addrspace(1)** %img1.addr, align 4
  %call = call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image1d_ro(%opencl.image1d_ro_t addrspace(1)* %0)
  %1 = load %opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)** %img2.addr, align 4
  %call1 = call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)* %1)
  %2 = load %opencl.image3d_ro_t addrspace(1)*, %opencl.image3d_ro_t addrspace(1)** %img3.addr, align 4
  %call2 = call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image3d_ro(%opencl.image3d_ro_t addrspace(1)* %2)
  %3 = load %opencl.image1d_array_ro_t addrspace(1)*, %opencl.image1d_array_ro_t addrspace(1)** %img4.addr, align 4
  %call3 = call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image1d_array_ro(%opencl.image1d_array_ro_t addrspace(1)* %3)
  %4 = load %opencl.image2d_array_ro_t addrspace(1)*, %opencl.image2d_array_ro_t addrspace(1)** %img5.addr, align 4
  %call4 = call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_array_ro(%opencl.image2d_array_ro_t addrspace(1)* %4)
  %5 = load %opencl.image2d_depth_ro_t addrspace(1)*, %opencl.image2d_depth_ro_t addrspace(1)** %img6.addr, align 4
  %call5 = call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_depth_ro(%opencl.image2d_depth_ro_t addrspace(1)* %5)
  %6 = load %opencl.image2d_array_depth_ro_t addrspace(1)*, %opencl.image2d_array_depth_ro_t addrspace(1)** %img7.addr, align 4
  %call6 = call spir_func i32 @_Z24get_image_num_mip_levels26ocl_image2d_array_depth_ro(%opencl.image2d_array_depth_ro_t addrspace(1)* %6)
  ret void
}

declare spir_func i32 @_Z24get_image_num_mip_levels14ocl_image1d_ro(%opencl.image1d_ro_t addrspace(1)*)

declare spir_func i32 @_Z24get_image_num_mip_levels14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)*)

declare spir_func i32 @_Z24get_image_num_mip_levels14ocl_image3d_ro(%opencl.image3d_ro_t addrspace(1)*)

declare spir_func i32 @_Z24get_image_num_mip_levels20ocl_image1d_array_ro(%opencl.image1d_array_ro_t addrspace(1)*)

declare spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_array_ro(%opencl.image2d_array_ro_t addrspace(1)*)

declare spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_depth_ro(%opencl.image2d_depth_ro_t addrspace(1)*)

declare spir_func i32 @_Z24get_image_num_mip_levels26ocl_image2d_array_depth_ro(%opencl.image2d_array_depth_ro_t addrspace(1)*)
