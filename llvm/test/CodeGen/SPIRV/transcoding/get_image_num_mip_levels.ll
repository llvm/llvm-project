; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

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

define spir_func void @testimage1d(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %img1, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img2, target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %img3, target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0) %img4, target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0) %img5, target("spirv.Image", void, 1, 1, 0, 0, 0, 0, 0) %img6, target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0) %img7) {
entry:
  %img1.addr = alloca target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), align 4
  %img2.addr = alloca target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), align 4
  %img3.addr = alloca target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), align 4
  %img4.addr = alloca target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0), align 4
  %img5.addr = alloca target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0), align 4
  %img6.addr = alloca target("spirv.Image", void, 1, 1, 0, 0, 0, 0, 0), align 4
  %img7.addr = alloca target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0), align 4
  store target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %img1, target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0)* %img1.addr, align 4
  store target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %img2, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)* %img2.addr, align 4
  store target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %img3, target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0)* %img3.addr, align 4
  store target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0) %img4, target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0)* %img4.addr, align 4
  store target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0) %img5, target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0)* %img5.addr, align 4
  store target("spirv.Image", void, 1, 1, 0, 0, 0, 0, 0) %img6, target("spirv.Image", void, 1, 1, 0, 0, 0, 0, 0)* %img6.addr, align 4
  store target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0) %img7, target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0)* %img7.addr, align 4
  %0 = load target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0)* %img1.addr, align 4
  %call = call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image1d_ro(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %0)
  %1 = load target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)* %img2.addr, align 4
  %call1 = call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %1)
  %2 = load target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0)* %img3.addr, align 4
  %call2 = call spir_func i32 @_Z24get_image_num_mip_levels14ocl_image3d_ro(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %2)
  %3 = load target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0), target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0)* %img4.addr, align 4
  %call3 = call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image1d_array_ro(target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0) %3)
  %4 = load target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0), target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0)* %img5.addr, align 4
  %call4 = call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0) %4)
  %5 = load target("spirv.Image", void, 1, 1, 0, 0, 0, 0, 0), target("spirv.Image", void, 1, 1, 0, 0, 0, 0, 0)* %img6.addr, align 4
  %call5 = call spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_depth_ro(target("spirv.Image", void, 1, 1, 0, 0, 0, 0, 0) %5)
  %6 = load target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0), target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0)* %img7.addr, align 4
  %call6 = call spir_func i32 @_Z24get_image_num_mip_levels26ocl_image2d_array_depth_ro(target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0) %6)
  ret void
}

declare spir_func i32 @_Z24get_image_num_mip_levels14ocl_image1d_ro(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0))

declare spir_func i32 @_Z24get_image_num_mip_levels14ocl_image2d_ro(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0))

declare spir_func i32 @_Z24get_image_num_mip_levels14ocl_image3d_ro(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0))

declare spir_func i32 @_Z24get_image_num_mip_levels20ocl_image1d_array_ro(target("spirv.Image", void, 0, 0, 1, 0, 0, 0, 0))

declare spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0))

declare spir_func i32 @_Z24get_image_num_mip_levels20ocl_image2d_depth_ro(target("spirv.Image", void, 1, 1, 0, 0, 0, 0, 0))

declare spir_func i32 @_Z24get_image_num_mip_levels26ocl_image2d_array_depth_ro(target("spirv.Image", void, 1, 1, 1, 0, 0, 0, 0))
