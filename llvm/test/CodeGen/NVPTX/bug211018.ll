; RUN: llc < %s | FileCheck %s

; Exercise the scheduler's long dependency path with a wide ordered reduction.
; https://github.com/llvm/llvm-project/issues/211018

target triple = "nvptx64-nvidia-cuda"

declare float @llvm.vector.reduce.fadd.v65536f32(float, <65536 x float>)

define void @kernel(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; Check that the final lane participates in the ordered reduction and is stored.
; CHECK-LABEL: .visible .func kernel(
; CHECK-DAG: ld.param.b64 [[OUT:%rd[0-9]+]], [kernel_param_0];
; CHECK-DAG: ld.param.b64 [[IN:%rd[0-9]+]], [kernel_param_1];
; CHECK: ld.global.v4.b32 {{{%r[0-9]+}}, {{%r[0-9]+}}, {{%r[0-9]+}}, [[LAST:%r[0-9]+]]}, [[[IN]]+262128];
; CHECK: mul.rn.f32 [[SQUARE:%r[0-9]+]], [[LAST]], [[LAST]];
; CHECK: add.rn.f32 [[RESULT:%r[0-9]+]], {{%r[0-9]+}}, [[SQUARE]];
; CHECK: st.global.b32 [[[OUT]]], [[RESULT]];
; CHECK: ret;
entry:
  %val = load <65536 x float>, ptr addrspace(1) %in, align 32
  %mul = fmul <65536 x float> %val, %val
  %res = call float @llvm.vector.reduce.fadd.v65536f32(
      float 0.000000e+00, <65536 x float> %mul)
  store float %res, ptr addrspace(1) %out, align 4
  ret void
}
