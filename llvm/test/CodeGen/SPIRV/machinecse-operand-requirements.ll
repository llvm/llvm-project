; RUN: llc -O2 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product,+SPV_INTEL_bindless_images %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product,+SPV_INTEL_bindless_images %s -o - -filetype=obj | spirv-val %}

; Check that comparisons, atomic stores, dot products and bindless image
; handle conversions compile at -O2 with the required capabilities and
; extensions. Computations repeated after a conditional block exercise
; MachineCSE PRE.

; CHECK-DAG: OpCapability Int64Atomics
; CHECK-DAG: OpCapability DotProduct
; CHECK-DAG: OpCapability DotProductInput4x8BitPacked
; CHECK-DAG: OpCapability BindlessImagesINTEL
; CHECK-DAG: OpExtension "SPV_KHR_integer_dot_product"
; CHECK-DAG: OpExtension "SPV_INTEL_bindless_images"
; CHECK: OpMemoryModel

; CHECK-LABEL: Begin function fcmp_pre
; CHECK: %[[#FV:]] = OpFMul
; CHECK: OpFOrdLessThan %[[#]] %[[#FV]]
define spir_kernel void @fcmp_pre(ptr addrspace(1) %p, ptr addrspace(1) %q, float %x, float %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit
then:
  %a = fmul float %x, %y
  store float %a, ptr addrspace(1) %p
  br label %exit
exit:
  %b = fmul float %x, %y
  %r = fcmp olt float %b, %y
  %e = zext i1 %r to i32
  store i32 %e, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: Begin function atomic_store_pre
; CHECK: %[[#AV:]] = OpIAdd
; CHECK: OpAtomicStore %[[#]] %[[#]] %[[#]] %[[#AV]]
define spir_kernel void @atomic_store_pre(ptr addrspace(1) %p, i64 %x, i64 %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit
then:
  %a = add i64 %x, %y
  store i64 %a, ptr addrspace(1) %p
  br label %exit
exit:
  %b = add i64 %x, %y
  store atomic i64 %b, ptr addrspace(1) %p seq_cst, align 8
  ret void
}

; CHECK-LABEL: Begin function dot_pre
; CHECK: %[[#DV:]] = OpIAdd
; CHECK: OpSDot %[[#]] %[[#DV]] %[[#]] PackedVectorFormat4x8Bit
define spir_kernel void @dot_pre(ptr addrspace(1) %p, i32 %acc, i32 %x, i32 %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit
then:
  %a = add i32 %x, %y
  store i32 %a, ptr addrspace(1) %p
  br label %exit
exit:
  %b = add i32 %x, %y
  %r = call i32 @llvm.spv.dot4add.i8packed(i32 %acc, i32 %b, i32 %y)
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: Begin function image_handle_pre
; CHECK: %[[#HV:]] = OpIAdd
; CHECK: OpConvertHandleToImageINTEL %[[#]] %[[#HV]]
define spir_func target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) @image_handle_pre(ptr addrspace(1) %p, i64 %x, i64 %y, i1 %c) {
entry:
  br i1 %c, label %then, label %exit
then:
  %a = add i64 %x, %y
  store i64 %a, ptr addrspace(1) %p
  br label %exit
exit:
  %b = add i64 %x, %y
  %img = call spir_func target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) @_Z33__spirv_ConvertHandleToImageINTELl(i64 %b)
  ret target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) %img
}

declare spir_func target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) @_Z33__spirv_ConvertHandleToImageINTELl(i64)
