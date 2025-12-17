; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; StorageImageReadWithoutFormat/StorageImageWriteWithoutFormat implicitly
; declare Shader, causing a SPIR-V module to be rejected by the OpenCL
; run-time. See https://github.com/KhronosGroup/SPIRV-Headers/issues/487
; De-facto, OpImageRead and OpImageWrite are allowed to use Unknown Image
; Formats when the Kernel capability is declared. We reflect this behavior
; in the test case, and leave the check under CHECK-SPIRV-NOT to track
; the issue and follow-up its final resolution when ready.
; CHECK-SPIRV-NOT: OpCapability StorageImageReadWithoutFormat

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

define dso_local spir_kernel void @kernelA(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input) {
entry:
  %input.addr = alloca target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), align 8
  %c = alloca <4 x i32>, align 16
  %.compoundliteral = alloca <4 x i32>, align 16
  store target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input, ptr %input.addr, align 8
  %0 = load target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), ptr %input.addr, align 8
  store <4 x i32> zeroinitializer, ptr %.compoundliteral, align 16
  %1 = load <4 x i32>, ptr %.compoundliteral, align 16
  %call = call spir_func <4 x i32> @_Z12read_imageui14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %0, <4 x i32> noundef %1)
  store <4 x i32> %call, ptr %c, align 16
  ret void
}

declare spir_func <4 x i32> @_Z12read_imageui14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %0, <4 x i32> noundef %1)

define dso_local spir_kernel void @kernelB(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input) {
entry:
  %input.addr = alloca target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), align 8
  %f = alloca <4 x float>, align 16
  %.compoundliteral = alloca <4 x i32>, align 16
  store target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input, ptr %input.addr, align 8
  %0 = load target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), ptr %input.addr, align 8
  store <4 x i32> zeroinitializer, ptr %.compoundliteral, align 16
  %1 = load <4 x i32>, ptr %.compoundliteral, align 16
  %call = call spir_func <4 x float> @_Z11read_imagef14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %0, <4 x i32> noundef %1)
  store <4 x float> %call, ptr %f, align 16
  ret void
}

declare spir_func <4 x float> @_Z11read_imagef14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %0, <4 x i32> noundef %1)
