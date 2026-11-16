; RUN: split-file %s %t
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-library %t/f16.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-library %t/f64.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-library %t/i16.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-library %t/bool.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-library %t/vector.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-library --spirv-ext=+SPV_EXT_shader_image_int64 %t/legal.ll -o - | FileCheck %s --check-prefix=LEGAL
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %t/opencl.ll -o - | FileCheck %s --check-prefix=OPENCL
; RUN: llc -O0 -mtriple=spirv-unknown-unknown-library %t/f16.ll -o - | FileCheck %s --check-prefix=GENERIC

; Vulkan requires the Sampled Type of every image, sampled or storage, to be a
; scalar 32-bit float, 32-bit integer or 64-bit integer.
; INVALID: OpTypeImage Sampled Type must be a scalar 32-bit float, 32-bit integer or 64-bit integer for Vulkan (VUID-StandaloneSpirv-OpTypeImage-04656)

; LEGAL-DAG: %[[F32:[0-9]+]] = OpTypeFloat 32
; LEGAL-DAG: %[[I32:[0-9]+]] = OpTypeInt 32 0
; LEGAL-DAG: %[[I64:[0-9]+]] = OpTypeInt 64 0
; LEGAL-DAG: OpTypeImage %[[F32]] 2D 0 0 0 1 Unknown
; LEGAL-DAG: OpTypeImage %[[I32]] Buffer 0 0 0 2 R32ui
; LEGAL-DAG: OpTypeImage %[[I64]] 2D 0 0 0 2 R64ui

; OpenCL images use a void Sampled Type.
; OPENCL: %[[VOID:[0-9]+]] = OpTypeVoid
; OPENCL: OpTypeImage %[[VOID]] 2D 0 0 0 0 Unknown ReadOnly

; The restriction is specific to Vulkan.
; GENERIC: %[[HALF:[0-9]+]] = OpTypeFloat 16
; GENERIC: OpTypeImage %[[HALF]] 2D 0 0 0 1 Unknown

;--- f16.ll
define spir_func void @sampled_f16(target("spirv.Image", half, 1, 0, 0, 0, 1, 0) %image) {
  ret void
}

;--- f64.ll
define spir_func void @storage_f64(target("spirv.Image", double, 1, 0, 0, 0, 2, 0) %image) {
  ret void
}

;--- i16.ll
define spir_func void @storage_i16(target("spirv.SignedImage", i16, 5, 0, 0, 0, 2, 0) %image) {
  ret void
}

;--- bool.ll
define spir_func void @sampled_bool(target("spirv.Image", i1, 5, 0, 0, 0, 1, 0) %image) {
  ret void
}

;--- vector.ll
define spir_func void @storage_vector(target("spirv.Image", <2 x float>, 1, 0, 0, 0, 2, 0) %image) {
  ret void
}

;--- legal.ll
define spir_func void @legal_images(target("spirv.Image", float, 1, 0, 0, 0, 1, 0) %sampled,
                                   target("spirv.Image", i32, 5, 0, 0, 0, 2, 33) %buffer,
                                   target("spirv.Image", i64, 1, 0, 0, 0, 2, 40) %wide) {
  ret void
}

;--- opencl.ll
define spir_func void @opencl_image(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %image) {
  ret void
}
