; An OpTypeImage with an R64ui/R64i Image Format requires the Int64ImageEXT
; capability and the SPV_EXT_shader_image_int64 extension.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_shader_image_int64 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_shader_image_int64 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Int64ImageEXT
; CHECK-DAG: OpExtension "SPV_EXT_shader_image_int64"
; CHECK-DAG: %[[#Int64Ty:]] = OpTypeInt 64 0
; R64ui Image Format is encoded as 40, R64i as 41.
; CHECK-DAG: %[[#ImgUTy:]] = OpTypeImage %[[#Int64Ty]] 2D 0 0 0 2 R64ui ReadWrite
; CHECK-DAG: %[[#ImgSTy:]] = OpTypeImage %[[#Int64Ty]] 2D 0 0 0 2 R64i ReadWrite

define spir_func void @foo(target("spirv.Image", i64, 1, 0, 0, 0, 2, 40, 2) %img) {
  ret void
}

define spir_func void @bar(target("spirv.Image", i64, 1, 0, 0, 0, 2, 41, 2) %img) {
  ret void
}
