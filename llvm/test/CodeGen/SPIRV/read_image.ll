; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: %[[#IntTy:]] = OpTypeInt
; CHECK-SPIRV: %[[#IVecTy:]] = OpTypeVector %[[#IntTy]]
; CHECK-SPIRV: %[[#FloatTy:]] = OpTypeFloat
; CHECK-SPIRV: %[[#FVecTy:]] = OpTypeVector %[[#FloatTy]]
; CHECK-SPIRV: OpImageRead %[[#IVecTy]]
; CHECK-SPIRV: OpImageRead %[[#FVecTy]]

;; __kernel void kernelA(__read_only image3d_t input) {
;;   uint4 c = read_imageui(input, (int4)(0, 0, 0, 0));
;; }
;;
;; __kernel void kernelB(__read_only image3d_t input) {
;;   float4 f = read_imagef(input, (int4)(0, 0, 0, 0));
;; }

%opencl.image3d_ro_t = type opaque

define dso_local spir_kernel void @kernelA(%opencl.image3d_ro_t addrspace(1)* %input) {
entry:
  %input.addr = alloca %opencl.image3d_ro_t addrspace(1)*, align 8
  %c = alloca <4 x i32>, align 16
  %.compoundliteral = alloca <4 x i32>, align 16
  store %opencl.image3d_ro_t addrspace(1)* %input, %opencl.image3d_ro_t addrspace(1)** %input.addr, align 8
  %0 = load %opencl.image3d_ro_t addrspace(1)*, %opencl.image3d_ro_t addrspace(1)** %input.addr, align 8
  store <4 x i32> zeroinitializer, <4 x i32>* %.compoundliteral, align 16
  %1 = load <4 x i32>, <4 x i32>* %.compoundliteral, align 16
  %call = call spir_func <4 x i32> @_Z12read_imageui14ocl_image3d_roDv4_i(%opencl.image3d_ro_t addrspace(1)* %0, <4 x i32> noundef %1)
  store <4 x i32> %call, <4 x i32>* %c, align 16
  ret void
}

declare spir_func <4 x i32> @_Z12read_imageui14ocl_image3d_roDv4_i(%opencl.image3d_ro_t addrspace(1)*, <4 x i32> noundef)

define dso_local spir_kernel void @kernelB(%opencl.image3d_ro_t addrspace(1)* %input) {
entry:
  %input.addr = alloca %opencl.image3d_ro_t addrspace(1)*, align 8
  %f = alloca <4 x float>, align 16
  %.compoundliteral = alloca <4 x i32>, align 16
  store %opencl.image3d_ro_t addrspace(1)* %input, %opencl.image3d_ro_t addrspace(1)** %input.addr, align 8
  %0 = load %opencl.image3d_ro_t addrspace(1)*, %opencl.image3d_ro_t addrspace(1)** %input.addr, align 8
  store <4 x i32> zeroinitializer, <4 x i32>* %.compoundliteral, align 16
  %1 = load <4 x i32>, <4 x i32>* %.compoundliteral, align 16
  %call = call spir_func <4 x float> @_Z11read_imagef14ocl_image3d_roDv4_i(%opencl.image3d_ro_t addrspace(1)* %0, <4 x i32> noundef %1)
  store <4 x float> %call, <4 x float>* %f, align 16
  ret void
}

declare spir_func <4 x float> @_Z11read_imagef14ocl_image3d_roDv4_i(%opencl.image3d_ro_t addrspace(1)*, <4 x i32> noundef)
