; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -stop-after=irtranslator -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: @__spirv_BuiltInLocalInvocationId = external addrspace(1) global ptr addrspace
; CHECK: define spir_kernel void @func
; CHECK-SAME: ptr addrspace(1) %{{.*}}

define spir_kernel void @func(ptr addrspace(1) %arg) {
entry:
  %x = call spir_func i64 @_Z12get_local_idj(i32 0)
  ret void
}

declare spir_func i64 @_Z12get_local_idj(i32)
