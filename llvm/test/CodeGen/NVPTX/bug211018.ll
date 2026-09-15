; RUN: llc < %s -filetype=null -o -

; Keep the issue reproducer as compile-only coverage. Compile-time comparisons
; are performed separately because a timing threshold would be host-dependent.
; https://github.com/llvm/llvm-project/issues/211018

target triple = "nvptx64-nvidia-cuda"

declare float @llvm.vector.reduce.fadd.v65536f32(float, <65536 x float>)

define void @kernel(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %val = load <65536 x float>, ptr addrspace(1) %in, align 32
  %mul = fmul <65536 x float> %val, %val
  %res = call float @llvm.vector.reduce.fadd.v65536f32(
      float 0.000000e+00, <65536 x float> %mul)
  store float %res, ptr addrspace(1) %out, align 4
  ret void
}
