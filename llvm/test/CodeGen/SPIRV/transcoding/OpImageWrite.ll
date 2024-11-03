; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: %[[#VOID_TY:]] = OpTypeVoid
; CHECK-SPIRV: %[[#IMG2D_WO_TY:]] = OpTypeImage %[[#VOID_TY]] 2D 0 0 0 0 Unknown WriteOnly
; CHECK-SPIRV: %[[#IMG2D_RW_TY:]] = OpTypeImage %[[#VOID_TY]] 2D 0 0 0 0 Unknown ReadWrite
; CHECK-SPIRV: %[[#IMG2D_ARRAY_WO_TY:]] = OpTypeImage %[[#VOID_TY]] 2D 0 1 0 0 Unknown WriteOnly
; CHECK-SPIRV: %[[#IMG2D_ARRAY_RW_TY:]] = OpTypeImage %[[#VOID_TY]] 2D 0 1 0 0 Unknown ReadWrite
; CHECK-SPIRV: %[[#IMG1D_WO_TY:]] = OpTypeImage %[[#VOID_TY]] 1D 0 0 0 0 Unknown WriteOnly
; CHECK-SPIRV: %[[#IMG1D_RW_TY:]] = OpTypeImage %[[#VOID_TY]] 1D 0 0 0 0 Unknown ReadWrite
; CHECK-SPIRV: %[[#IMG1D_BUFFER_WO_TY:]] = OpTypeImage %[[#VOID_TY]] Buffer 0 0 0 0 Unknown WriteOnly
; CHECK-SPIRV: %[[#IMG1D_BUFFER_RW_TY:]] = OpTypeImage %[[#VOID_TY]] Buffer 0 0 0 0 Unknown ReadWrite
; CHECK-SPIRV: %[[#IMG1D_ARRAY_WO_TY:]] = OpTypeImage %[[#VOID_TY]] 1D 0 1 0 0 Unknown WriteOnly
; CHECK-SPIRV: %[[#IMG1D_ARRAY_RW_TY:]] = OpTypeImage %[[#VOID_TY]] 1D 0 1 0 0 Unknown ReadWrite
; CHECK-SPIRV: %[[#IMG2D_DEPTH_WO_TY:]] = OpTypeImage %[[#VOID_TY]] 2D 1 0 0 0 Unknown WriteOnly
; CHECK-SPIRV: %[[#IMG2D_ARRAY_DEPTH_WO_TY:]] = OpTypeImage %[[#VOID_TY]] 2D 1 1 0 0 Unknown WriteOnly
; CHECK-SPIRV: %[[#IMG3D_WO_TY:]] = OpTypeImage %[[#VOID_TY]] 3D 0 0 0 0 Unknown WriteOnly
; CHECK-SPIRV: %[[#IMG3D_RW_TY:]] = OpTypeImage %[[#VOID_TY]] 3D 0 0 0 0 Unknown ReadWrite

%opencl.image2d_wo_t = type opaque
%opencl.image2d_rw_t = type opaque
%opencl.image2d_array_wo_t = type opaque
%opencl.image2d_array_rw_t = type opaque
%opencl.image1d_wo_t = type opaque
%opencl.image1d_rw_t = type opaque
%opencl.image1d_buffer_wo_t = type opaque
%opencl.image1d_buffer_rw_t = type opaque
%opencl.image1d_array_wo_t = type opaque
%opencl.image1d_array_rw_t = type opaque
%opencl.image2d_depth_wo_t = type opaque
%opencl.image2d_array_depth_wo_t = type opaque
%opencl.image3d_wo_t = type opaque
%opencl.image3d_rw_t = type opaque

;; kernel void test_img2d(write_only image2d_t image_wo, read_write image2d_t image_rw)
;; {
;;     write_imagef(image_wo, (int2)(0,0), (float4)(0,0,0,0));
;;     write_imagei(image_wo, (int2)(0,0), (int4)(0,0,0,0));
;;     write_imagef(image_rw, (int2)(0,0), (float4)(0,0,0,0));
;;     write_imagei(image_rw, (int2)(0,0), (int4)(0,0,0,0));
;;
    ;; LOD
;;     write_imagef(image_wo, (int2)(0,0), 0, (float4)(0,0,0,0));
;;     write_imagei(image_wo, (int2)(0,0), 0, (int4)(0,0,0,0));
;; }

; CHECK-SPIRV: %[[#IMG2D_WO:]] = OpFunctionParameter %[[#IMG2D_WO_TY]]
; CHECK-SPIRV: %[[#IMG2D_RW:]] = OpFunctionParameter %[[#IMG2D_RW_TY]]

; CHECK-SPIRV: OpImageWrite %[[#IMG2D_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_RW]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_RW]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_WO]]

define dso_local spir_kernel void @test_img2d(%opencl.image2d_wo_t addrspace(1)* %image_wo, %opencl.image2d_rw_t addrspace(1)* %image_rw) local_unnamed_addr {
entry:
  call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(%opencl.image2d_wo_t addrspace(1)* %image_wo, <2 x i32> noundef zeroinitializer, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei14ocl_image2d_woDv2_iDv4_i(%opencl.image2d_wo_t addrspace(1)* %image_wo, <2 x i32> noundef zeroinitializer, <4 x i32> noundef zeroinitializer)
  call spir_func void @_Z12write_imagef14ocl_image2d_rwDv2_iDv4_f(%opencl.image2d_rw_t addrspace(1)* %image_rw, <2 x i32> noundef zeroinitializer, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei14ocl_image2d_rwDv2_iDv4_i(%opencl.image2d_rw_t addrspace(1)* %image_rw, <2 x i32> noundef zeroinitializer, <4 x i32> noundef zeroinitializer)
  call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iiDv4_f(%opencl.image2d_wo_t addrspace(1)* %image_wo, <2 x i32> noundef zeroinitializer, i32 noundef 0, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei14ocl_image2d_woDv2_iiDv4_i(%opencl.image2d_wo_t addrspace(1)* %image_wo, <2 x i32> noundef zeroinitializer, i32 noundef 0, <4 x i32> noundef zeroinitializer)
  ret void
}

declare spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(%opencl.image2d_wo_t addrspace(1)*, <2 x i32> noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei14ocl_image2d_woDv2_iDv4_i(%opencl.image2d_wo_t addrspace(1)*, <2 x i32> noundef, <4 x i32> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef14ocl_image2d_rwDv2_iDv4_f(%opencl.image2d_rw_t addrspace(1)*, <2 x i32> noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei14ocl_image2d_rwDv2_iDv4_i(%opencl.image2d_rw_t addrspace(1)*, <2 x i32> noundef, <4 x i32> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iiDv4_f(%opencl.image2d_wo_t addrspace(1)*, <2 x i32> noundef, i32 noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei14ocl_image2d_woDv2_iiDv4_i(%opencl.image2d_wo_t addrspace(1)*, <2 x i32> noundef, i32 noundef, <4 x i32> noundef) local_unnamed_addr

;; kernel void test_img2d_array(write_only image2d_array_t image_wo, read_write image2d_array_t image_rw)
;; {
;;     write_imagef(image_wo, (int4)(0,0,0,0), (float4)(0,0,0,0));
;;     write_imagei(image_wo, (int4)(0,0,0,0), (int4)(0,0,0,0));
;;     write_imagef(image_rw, (int4)(0,0,0,0), (float4)(0,0,0,0));
;;     write_imagei(image_rw, (int4)(0,0,0,0), (int4)(0,0,0,0));
;;
    ;; LOD
;;     write_imagef(image_wo, (int4)(0,0,0,0), 0, (float4)(0,0,0,0));
;;     write_imagei(image_wo, (int4)(0,0,0,0), 0, (int4)(0,0,0,0));
;; }

; CHECK-SPIRV: %[[#IMG2D_ARRAY_WO:]] = OpFunctionParameter %[[#IMG2D_ARRAY_WO_TY]]
; CHECK-SPIRV: %[[#IMG2D_ARRAY_RW:]] = OpFunctionParameter %[[#IMG2D_ARRAY_RW_TY]]

; CHECK-SPIRV: OpImageWrite %[[#IMG2D_ARRAY_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_ARRAY_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_ARRAY_RW]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_ARRAY_RW]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_ARRAY_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_ARRAY_WO]]

define dso_local spir_kernel void @test_img2d_array(%opencl.image2d_array_wo_t addrspace(1)* %image_wo, %opencl.image2d_array_rw_t addrspace(1)* %image_rw) local_unnamed_addr {
entry:
  call spir_func void @_Z12write_imagef20ocl_image2d_array_woDv4_iDv4_f(%opencl.image2d_array_wo_t addrspace(1)* %image_wo, <4 x i32> noundef zeroinitializer, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei20ocl_image2d_array_woDv4_iS0_(%opencl.image2d_array_wo_t addrspace(1)* %image_wo, <4 x i32> noundef zeroinitializer, <4 x i32> noundef zeroinitializer)
  call spir_func void @_Z12write_imagef20ocl_image2d_array_rwDv4_iDv4_f(%opencl.image2d_array_rw_t addrspace(1)* %image_rw, <4 x i32> noundef zeroinitializer, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei20ocl_image2d_array_rwDv4_iS0_(%opencl.image2d_array_rw_t addrspace(1)* %image_rw, <4 x i32> noundef zeroinitializer, <4 x i32> noundef zeroinitializer)
  call spir_func void @_Z12write_imagef20ocl_image2d_array_woDv4_iiDv4_f(%opencl.image2d_array_wo_t addrspace(1)* %image_wo, <4 x i32> noundef zeroinitializer, i32 noundef 0, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei20ocl_image2d_array_woDv4_iiS0_(%opencl.image2d_array_wo_t addrspace(1)* %image_wo, <4 x i32> noundef zeroinitializer, i32 noundef 0, <4 x i32> noundef zeroinitializer)
  ret void
}

declare spir_func void @_Z12write_imagef20ocl_image2d_array_woDv4_iDv4_f(%opencl.image2d_array_wo_t addrspace(1)*, <4 x i32> noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei20ocl_image2d_array_woDv4_iS0_(%opencl.image2d_array_wo_t addrspace(1)*, <4 x i32> noundef, <4 x i32> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef20ocl_image2d_array_rwDv4_iDv4_f(%opencl.image2d_array_rw_t addrspace(1)*, <4 x i32> noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei20ocl_image2d_array_rwDv4_iS0_(%opencl.image2d_array_rw_t addrspace(1)*, <4 x i32> noundef, <4 x i32> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef20ocl_image2d_array_woDv4_iiDv4_f(%opencl.image2d_array_wo_t addrspace(1)*, <4 x i32> noundef, i32 noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei20ocl_image2d_array_woDv4_iiS0_(%opencl.image2d_array_wo_t addrspace(1)*, <4 x i32> noundef, i32 noundef, <4 x i32> noundef) local_unnamed_addr

;; kernel void test_img1d(write_only image1d_t image_wo, read_write image1d_t image_rw)
;; {
;;     write_imagef(image_wo, 0, (float4)(0,0,0,0));
;;     write_imagei(image_wo, 0, (int4)(0,0,0,0));
;;     write_imagef(image_rw, 0, (float4)(0,0,0,0));
;;     write_imagei(image_rw, 0, (int4)(0,0,0,0));
;;
    ;; LOD
;;     write_imagef(image_wo, 0, 0, (float4)(0,0,0,0));
;;     write_imagei(image_wo, 0, 0, (int4)(0,0,0,0));
;; }

; CHECK-SPIRV: %[[#IMG1D_WO:]] = OpFunctionParameter %[[#IMG1D_WO_TY]]
; CHECK-SPIRV: %[[#IMG1D_RW:]] = OpFunctionParameter %[[#IMG1D_RW_TY]]

; CHECK-SPIRV: OpImageWrite %[[#IMG1D_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_RW]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_RW]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_WO]]

define dso_local spir_kernel void @test_img1d(%opencl.image1d_wo_t addrspace(1)* %image_wo, %opencl.image1d_rw_t addrspace(1)* %image_rw) local_unnamed_addr {
entry:
  call spir_func void @_Z12write_imagef14ocl_image1d_woiDv4_f(%opencl.image1d_wo_t addrspace(1)* %image_wo, i32 noundef 0, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei14ocl_image1d_woiDv4_i(%opencl.image1d_wo_t addrspace(1)* %image_wo, i32 noundef 0, <4 x i32> noundef zeroinitializer)
  call spir_func void @_Z12write_imagef14ocl_image1d_rwiDv4_f(%opencl.image1d_rw_t addrspace(1)* %image_rw, i32 noundef 0, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei14ocl_image1d_rwiDv4_i(%opencl.image1d_rw_t addrspace(1)* %image_rw, i32 noundef 0, <4 x i32> noundef zeroinitializer)
  call spir_func void @_Z12write_imagef14ocl_image1d_woiiDv4_f(%opencl.image1d_wo_t addrspace(1)* %image_wo, i32 noundef 0, i32 noundef 0, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei14ocl_image1d_woiiDv4_i(%opencl.image1d_wo_t addrspace(1)* %image_wo, i32 noundef 0, i32 noundef 0, <4 x i32> noundef zeroinitializer)
  ret void
}

declare spir_func void @_Z12write_imagef14ocl_image1d_woiDv4_f(%opencl.image1d_wo_t addrspace(1)*, i32 noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei14ocl_image1d_woiDv4_i(%opencl.image1d_wo_t addrspace(1)*, i32 noundef, <4 x i32> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef14ocl_image1d_rwiDv4_f(%opencl.image1d_rw_t addrspace(1)*, i32 noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei14ocl_image1d_rwiDv4_i(%opencl.image1d_rw_t addrspace(1)*, i32 noundef, <4 x i32> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef14ocl_image1d_woiiDv4_f(%opencl.image1d_wo_t addrspace(1)*, i32 noundef, i32 noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei14ocl_image1d_woiiDv4_i(%opencl.image1d_wo_t addrspace(1)*, i32 noundef, i32 noundef, <4 x i32> noundef) local_unnamed_addr

;; kernel void test_img1d_buffer(write_only image1d_buffer_t image_wo, read_write image1d_buffer_t image_rw)
;; {
;;     write_imagef(image_wo, 0, (float4)(0,0,0,0));
;;     write_imagei(image_wo, 0, (int4)(0,0,0,0));
;;     write_imagef(image_rw, 0, (float4)(0,0,0,0));
;;     write_imagei(image_rw, 0, (int4)(0,0,0,0));
;; }

; CHECK-SPIRV: %[[#IMG1D_BUFFER_WO:]] = OpFunctionParameter %[[#IMG1D_BUFFER_WO_TY]]
; CHECK-SPIRV: %[[#IMG1D_BUFFER_RW:]] = OpFunctionParameter %[[#IMG1D_BUFFER_RW_TY]]

; CHECK-SPIRV: OpImageWrite %[[#IMG1D_BUFFER_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_BUFFER_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_BUFFER_RW]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_BUFFER_RW]]

define dso_local spir_kernel void @test_img1d_buffer(%opencl.image1d_buffer_wo_t addrspace(1)* %image_wo, %opencl.image1d_buffer_rw_t addrspace(1)* %image_rw) local_unnamed_addr {
entry:
  call spir_func void @_Z12write_imagef21ocl_image1d_buffer_woiDv4_f(%opencl.image1d_buffer_wo_t addrspace(1)* %image_wo, i32 noundef 0, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei21ocl_image1d_buffer_woiDv4_i(%opencl.image1d_buffer_wo_t addrspace(1)* %image_wo, i32 noundef 0, <4 x i32> noundef zeroinitializer)
  call spir_func void @_Z12write_imagef21ocl_image1d_buffer_rwiDv4_f(%opencl.image1d_buffer_rw_t addrspace(1)* %image_rw, i32 noundef 0, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei21ocl_image1d_buffer_rwiDv4_i(%opencl.image1d_buffer_rw_t addrspace(1)* %image_rw, i32 noundef 0, <4 x i32> noundef zeroinitializer)
  ret void
}

declare spir_func void @_Z12write_imagef21ocl_image1d_buffer_woiDv4_f(%opencl.image1d_buffer_wo_t addrspace(1)*, i32 noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei21ocl_image1d_buffer_woiDv4_i(%opencl.image1d_buffer_wo_t addrspace(1)*, i32 noundef, <4 x i32> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef21ocl_image1d_buffer_rwiDv4_f(%opencl.image1d_buffer_rw_t addrspace(1)*, i32 noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei21ocl_image1d_buffer_rwiDv4_i(%opencl.image1d_buffer_rw_t addrspace(1)*, i32 noundef, <4 x i32> noundef) local_unnamed_addr

;; kernel void test_img1d_array(write_only image1d_array_t image_wo, read_write image1d_array_t image_rw)
;; {
;;     write_imagef(image_wo, (int2)(0,0), (float4)(0,0,0,0));
;;     write_imagei(image_wo, (int2)(0,0), (int4)(0,0,0,0));
;;     write_imagef(image_rw, (int2)(0,0), (float4)(0,0,0,0));
;;     write_imagei(image_rw, (int2)(0,0), (int4)(0,0,0,0));
;;
    ;; LOD
;;     write_imagef(image_wo, (int2)(0,0), 0, (float4)(0,0,0,0));
;;     write_imagei(image_wo, (int2)(0,0), 0, (int4)(0,0,0,0));
;; }

; CHECK-SPIRV: %[[#IMG1D_ARRAY_WO:]] = OpFunctionParameter %[[#IMG1D_ARRAY_WO_TY]]
; CHECK-SPIRV: %[[#IMG1D_ARRAY_RW:]] = OpFunctionParameter %[[#IMG1D_ARRAY_RW_TY]]

; CHECK-SPIRV: OpImageWrite %[[#IMG1D_ARRAY_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_ARRAY_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_ARRAY_RW]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_ARRAY_RW]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_ARRAY_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG1D_ARRAY_WO]]

define dso_local spir_kernel void @test_img1d_array(%opencl.image1d_array_wo_t addrspace(1)* %image_wo, %opencl.image1d_array_rw_t addrspace(1)* %image_rw) local_unnamed_addr {
entry:
  call spir_func void @_Z12write_imagef20ocl_image1d_array_woDv2_iDv4_f(%opencl.image1d_array_wo_t addrspace(1)* %image_wo, <2 x i32> noundef zeroinitializer, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei20ocl_image1d_array_woDv2_iDv4_i(%opencl.image1d_array_wo_t addrspace(1)* %image_wo, <2 x i32> noundef zeroinitializer, <4 x i32> noundef zeroinitializer)
  call spir_func void @_Z12write_imagef20ocl_image1d_array_rwDv2_iDv4_f(%opencl.image1d_array_rw_t addrspace(1)* %image_rw, <2 x i32> noundef zeroinitializer, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei20ocl_image1d_array_rwDv2_iDv4_i(%opencl.image1d_array_rw_t addrspace(1)* %image_rw, <2 x i32> noundef zeroinitializer, <4 x i32> noundef zeroinitializer)
  call spir_func void @_Z12write_imagef20ocl_image1d_array_woDv2_iiDv4_f(%opencl.image1d_array_wo_t addrspace(1)* %image_wo, <2 x i32> noundef zeroinitializer, i32 noundef 0, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei20ocl_image1d_array_woDv2_iiDv4_i(%opencl.image1d_array_wo_t addrspace(1)* %image_wo, <2 x i32> noundef zeroinitializer, i32 noundef 0, <4 x i32> noundef zeroinitializer)
  ret void
}

declare spir_func void @_Z12write_imagef20ocl_image1d_array_woDv2_iDv4_f(%opencl.image1d_array_wo_t addrspace(1)*, <2 x i32> noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei20ocl_image1d_array_woDv2_iDv4_i(%opencl.image1d_array_wo_t addrspace(1)*, <2 x i32> noundef, <4 x i32> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef20ocl_image1d_array_rwDv2_iDv4_f(%opencl.image1d_array_rw_t addrspace(1)*, <2 x i32> noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei20ocl_image1d_array_rwDv2_iDv4_i(%opencl.image1d_array_rw_t addrspace(1)*, <2 x i32> noundef, <4 x i32> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef20ocl_image1d_array_woDv2_iiDv4_f(%opencl.image1d_array_wo_t addrspace(1)*, <2 x i32> noundef, i32 noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei20ocl_image1d_array_woDv2_iiDv4_i(%opencl.image1d_array_wo_t addrspace(1)*, <2 x i32> noundef, i32 noundef, <4 x i32> noundef) local_unnamed_addr

;; kernel void test_img2d_depth(write_only image2d_depth_t image_wo)
;; {
;;     write_imagef(image_wo, (int2)(0,0), (float)(0));
;;     write_imagef(image_wo, (int2)(0,0), (float)(0));
;;
    ;; LOD
;;     write_imagef(image_wo, (int2)(0,0), 0, (float)(0));
;; }

; CHECK-SPIRV: %[[#IMG2D_DEPTH_WO:]] = OpFunctionParameter %[[#IMG2D_DEPTH_WO_TY]]

; CHECK-SPIRV: OpImageWrite %[[#IMG2D_DEPTH_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_DEPTH_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_DEPTH_WO]]

define dso_local spir_kernel void @test_img2d_depth(%opencl.image2d_depth_wo_t addrspace(1)* %image_wo) local_unnamed_addr {
entry:
  call spir_func void @_Z12write_imagef20ocl_image2d_depth_woDv2_if(%opencl.image2d_depth_wo_t addrspace(1)* %image_wo, <2 x i32> noundef zeroinitializer, float noundef 0.000000e+00)
  call spir_func void @_Z12write_imagef20ocl_image2d_depth_woDv2_if(%opencl.image2d_depth_wo_t addrspace(1)* %image_wo, <2 x i32> noundef zeroinitializer, float noundef 0.000000e+00)
  call spir_func void @_Z12write_imagef20ocl_image2d_depth_woDv2_iif(%opencl.image2d_depth_wo_t addrspace(1)* %image_wo, <2 x i32> noundef zeroinitializer, i32 noundef 0, float noundef 0.000000e+00)
  ret void
}

declare spir_func void @_Z12write_imagef20ocl_image2d_depth_woDv2_if(%opencl.image2d_depth_wo_t addrspace(1)*, <2 x i32> noundef, float noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef20ocl_image2d_depth_woDv2_iif(%opencl.image2d_depth_wo_t addrspace(1)*, <2 x i32> noundef, i32 noundef, float noundef) local_unnamed_addr

;; kernel void test_img2d_array_depth(write_only image2d_array_depth_t image_wo)
;; {
;;     write_imagef(image_wo, (int4)(0,0,0,0), (float)(0));
;;     write_imagef(image_wo, (int4)(0,0,0,0), (float)(0));
;;
    ;; LOD
;;     write_imagef(image_wo, (int4)(0,0,0,0), 0, (float)(0));
;; }

; CHECK-SPIRV: %[[#IMG2D_ARRAY_DEPTH_WO:]] = OpFunctionParameter %[[#IMG2D_ARRAY_DEPTH_WO_TY]]

; CHECK-SPIRV: OpImageWrite %[[#IMG2D_ARRAY_DEPTH_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_ARRAY_DEPTH_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG2D_ARRAY_DEPTH_WO]]

define dso_local spir_kernel void @test_img2d_array_depth(%opencl.image2d_array_depth_wo_t addrspace(1)* %image_wo) local_unnamed_addr {
entry:
  call spir_func void @_Z12write_imagef26ocl_image2d_array_depth_woDv4_if(%opencl.image2d_array_depth_wo_t addrspace(1)* %image_wo, <4 x i32> noundef zeroinitializer, float noundef 0.000000e+00)
  call spir_func void @_Z12write_imagef26ocl_image2d_array_depth_woDv4_if(%opencl.image2d_array_depth_wo_t addrspace(1)* %image_wo, <4 x i32> noundef zeroinitializer, float noundef 0.000000e+00)
  call spir_func void @_Z12write_imagef26ocl_image2d_array_depth_woDv4_iif(%opencl.image2d_array_depth_wo_t addrspace(1)* %image_wo, <4 x i32> noundef zeroinitializer, i32 noundef 0, float noundef 0.000000e+00)
  ret void
}

declare spir_func void @_Z12write_imagef26ocl_image2d_array_depth_woDv4_if(%opencl.image2d_array_depth_wo_t addrspace(1)*, <4 x i32> noundef, float noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef26ocl_image2d_array_depth_woDv4_iif(%opencl.image2d_array_depth_wo_t addrspace(1)*, <4 x i32> noundef, i32 noundef, float noundef) local_unnamed_addr

;; kernel void test_img3d(write_only image3d_t image_wo, read_write image3d_t image_rw)
;; {
;;     write_imagef(image_wo, (int4)(0,0,0,0), (float4)(0,0,0,0));
;;     write_imagei(image_wo, (int4)(0,0,0,0), (int4)(0,0,0,0));
;;     write_imagef(image_rw, (int4)(0,0,0,0), (float4)(0,0,0,0));
;;     write_imagei(image_rw, (int4)(0,0,0,0), (int4)(0,0,0,0));
;;
    ;; LOD
;;     write_imagef(image_wo, (int4)(0,0,0,0), 0, (float4)(0,0,0,0));
;;     write_imagei(image_wo, (int4)(0,0,0,0), 0, (int4)(0,0,0,0));
;; }

; CHECK-SPIRV: %[[#IMG3D_WO:]] = OpFunctionParameter %[[#IMG3D_WO_TY]]
; CHECK-SPIRV: %[[#IMG3D_RW:]] = OpFunctionParameter %[[#IMG3D_RW_TY]]

; CHECK-SPIRV: OpImageWrite %[[#IMG3D_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG3D_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG3D_RW]]
; CHECK-SPIRV: OpImageWrite %[[#IMG3D_RW]]
; CHECK-SPIRV: OpImageWrite %[[#IMG3D_WO]]
; CHECK-SPIRV: OpImageWrite %[[#IMG3D_WO]]

define dso_local spir_kernel void @test_img3d(%opencl.image3d_wo_t addrspace(1)* %image_wo, %opencl.image3d_rw_t addrspace(1)* %image_rw) local_unnamed_addr {
entry:
  call spir_func void @_Z12write_imagef14ocl_image3d_woDv4_iDv4_f(%opencl.image3d_wo_t addrspace(1)* %image_wo, <4 x i32> noundef zeroinitializer, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei14ocl_image3d_woDv4_iS0_(%opencl.image3d_wo_t addrspace(1)* %image_wo, <4 x i32> noundef zeroinitializer, <4 x i32> noundef zeroinitializer)
  call spir_func void @_Z12write_imagef14ocl_image3d_rwDv4_iDv4_f(%opencl.image3d_rw_t addrspace(1)* %image_rw, <4 x i32> noundef zeroinitializer, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei14ocl_image3d_rwDv4_iS0_(%opencl.image3d_rw_t addrspace(1)* %image_rw, <4 x i32> noundef zeroinitializer, <4 x i32> noundef zeroinitializer)
  call spir_func void @_Z12write_imagef14ocl_image3d_woDv4_iiDv4_f(%opencl.image3d_wo_t addrspace(1)* %image_wo, <4 x i32> noundef zeroinitializer, i32 noundef 0, <4 x float> noundef zeroinitializer)
  call spir_func void @_Z12write_imagei14ocl_image3d_woDv4_iiS0_(%opencl.image3d_wo_t addrspace(1)* %image_wo, <4 x i32> noundef zeroinitializer, i32 noundef 0, <4 x i32> noundef zeroinitializer)
  ret void
}

declare spir_func void @_Z12write_imagef14ocl_image3d_woDv4_iDv4_f(%opencl.image3d_wo_t addrspace(1)*, <4 x i32> noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei14ocl_image3d_woDv4_iS0_(%opencl.image3d_wo_t addrspace(1)*, <4 x i32> noundef, <4 x i32> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef14ocl_image3d_rwDv4_iDv4_f(%opencl.image3d_rw_t addrspace(1)*, <4 x i32> noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei14ocl_image3d_rwDv4_iS0_(%opencl.image3d_rw_t addrspace(1)*, <4 x i32> noundef, <4 x i32> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagef14ocl_image3d_woDv4_iiDv4_f(%opencl.image3d_wo_t addrspace(1)*, <4 x i32> noundef, i32 noundef, <4 x float> noundef) local_unnamed_addr

declare spir_func void @_Z12write_imagei14ocl_image3d_woDv4_iiS0_(%opencl.image3d_wo_t addrspace(1)*, <4 x i32> noundef, i32 noundef, <4 x i32> noundef) local_unnamed_addr
