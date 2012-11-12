; RUN: llc < %s -march=nvptx -mcpu=sm_10 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_10 | FileCheck %s
; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s


;; Kernel function using ptx_kernel calling conv

; CHECK: .entry kernel_func
define ptx_kernel void @kernel_func(float* %a) {
; CHECK: ret
  ret void
}

;; Device function
; CHECK: .func device_func
define void @device_func(float* %a) {
; CHECK: ret
  ret void
}

;; Kernel function using NVVM metadata
; CHECK: .entry metadata_kernel
define void @metadata_kernel(float* %a) {
; CHECK: ret
  ret void
}


!nvvm.annotations = !{!1}

!1 = metadata !{void (float*)* @metadata_kernel, metadata !"kernel", i32 1}
