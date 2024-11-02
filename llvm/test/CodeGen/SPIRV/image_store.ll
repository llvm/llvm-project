; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

;; Image types may be represented in two ways while translating to SPIR-V:
;; - OpenCL form, for example, '%opencl.image2d_ro_t',
;; - SPIR-V form, for example, '%spirv.Image._void_1_0_0_0_0_0_0',
;; but it is still one type which should be translated to one SPIR-V type.
;;
;; The test checks that the code below is successfully translated and only one
;; SPIR-V type for images is generated.

; CHECK:     OpTypeImage
; CHECK-NOT: OpTypeImage

%opencl.image2d_ro_t = type opaque
%spirv.Image._void_1_0_0_0_0_0_0 = type opaque

define spir_kernel void @read_image(%opencl.image2d_ro_t addrspace(1)* %srcimg) {
entry:
  %srcimg.addr = alloca %opencl.image2d_ro_t addrspace(1)*, align 8
  %spirvimg.addr = alloca %spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)*, align 8
  store %opencl.image2d_ro_t addrspace(1)* %srcimg, %opencl.image2d_ro_t addrspace(1)** %srcimg.addr, align 8
  ret void
}
