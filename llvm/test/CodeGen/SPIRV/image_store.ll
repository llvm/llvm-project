; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

;; Image types may be represented in two ways while translating to SPIR-V:
;; - OpenCL form based on pointers-to-opaque-structs, e.g. '%opencl.image2d_ro_t',
;; - SPIR-V form based on TargetExtType, e.g. 'target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0)',
;; but it is still one type which should be translated to one SPIR-V type.
;;
;; The test checks that the code below is successfully translated and only one
;; SPIR-V type for images is generated.

; CHECK:     OpTypeImage
; CHECK-NOT: OpTypeImage

%opencl.image2d_ro_t = type opaque

define spir_kernel void @read_image(%opencl.image2d_ro_t addrspace(1)* %srcimg) {
entry:
  %srcimg.addr = alloca %opencl.image2d_ro_t addrspace(1)*, align 8
  %spirvimg.addr = alloca target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), align 8
  store %opencl.image2d_ro_t addrspace(1)* %srcimg, %opencl.image2d_ro_t addrspace(1)** %srcimg.addr, align 8
  ret void
}
