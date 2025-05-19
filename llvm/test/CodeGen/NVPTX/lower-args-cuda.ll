; RUN: not llc < %s -mcpu=sm_75  -o /dev/null 2>&1 | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Make sure we exit with an error message for this input, as pointers to the
; shared address-space are only supported as kernel args in NVCL, not CUDA.
; CHECK:  .shared ptr kernel args unsupported in CUDA.
define ptx_kernel void @ptr_nongeneric(ptr addrspace(1) %out, ptr addrspace(3) %in) {
  %v = load i32, ptr addrspace(3) %in, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
