; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_side_avc_motion_estimation %s -o - | FileCheck %s
; XFAIL: *

; CHECK: OpName %[[#Name:]] "_Z31intel_sub_group_avc_mce_ime_boo"
; CHECK: %[[#]] = OpFunctionCall %[[#]] %[[#Name]]

define spir_func void @foo() {
entry:
  call spir_func void @_Z31intel_sub_group_avc_mce_ime_boo()
  ret void
}
declare spir_func void @_Z31intel_sub_group_avc_mce_ime_boo()
