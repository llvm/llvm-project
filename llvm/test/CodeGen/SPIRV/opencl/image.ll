; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; FIXME: Write tests to ensure invalid usage of image are rejected, such as:
;;  - invalid AS (only global is allowed);
;;  - used in struct, union, array, pointer or return types;
;;  - used with invalid CV-qualifiers (const or volatile in C99).
;; FIXME: Write further tests to cover _array, _buffer, _depth, ... types.

define void @foo(
  target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %a,
  target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %b,
  target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 2) %c,
  i32 addrspace(1)* %d
) {
  %pixel = call <4 x i32> @_Z11read_imagei14ocl_image1d_roi(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %a, i32 0)
  call void @_Z12write_imagei14ocl_image2d_woDv2_iDv4_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %b, <2 x i32> zeroinitializer, <4 x i32> %pixel)
  %size = call i32 @_Z15get_image_width14ocl_image3d_rw(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 2) %c)
  store i32 %size, i32 addrspace(1)* %d
  ret void
}

declare <4 x i32> @_Z11read_imagei14ocl_image1d_roi(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), i32)

declare void @_Z12write_imagei14ocl_image2d_woDv2_iDv4_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, <4 x i32>)

declare i32 @_Z15get_image_width14ocl_image3d_rw(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 2))


;; Capabilities:
; CHECK-DAG: OpCapability ImageReadWrite
; CHECK-NOT: DAG-FENCE

;; Types, Constants and Variables:
;; FIXME: The values should be double checked here.
; CHECK-DAG: %[[#IMG_1D:]] = OpTypeImage %[[#VOID:]] 1D 0 0 0 0 Unknown ReadOnly
; CHECK-DAG: %[[#IMG_2D:]] = OpTypeImage %[[#VOID]] 2D 0 0 0 0 Unknown WriteOnly
; CHECK-DAG: %[[#IMG_3D:]] = OpTypeImage %[[#VOID]] 3D 0 0 0 0 Unknown ReadWrite
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#I32:]]
; CHECK-DAG: %[[#FN:]] = OpTypeFunction %[[#VOID]] %[[#IMG_1D]] %[[#IMG_2D]] %[[#IMG_3D]] %[[#PTR]]

;; Functions:
; CHECK: OpFunction %[[#VOID]] None %[[#FN]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#IMG_1D]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#IMG_2D]]
; CHECK: %[[#C:]] = OpFunctionParameter %[[#IMG_3D]]
; CHECK: %[[#D:]] = OpFunctionParameter %[[#PTR]]
; CHECK: %[[#PIXEL:]] = OpImageRead %[[#VEC:]] %[[#A]] %[[#]]
; CHECK: OpImageWrite %[[#B]] %[[#]] %[[#PIXEL]]
;; FIXME: It is unclear which of OpImageQuerySize and OpImageQuerySizeLod should be used.
; CHECK: %[[#SIZE:]] = OpImageQuerySize{{(Lod)?}} %[[#]] %[[#C]]
