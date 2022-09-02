;; Make sure backend doesn't crash if the program contains
;; a mangled function which is not an OpenCL bultin.
;; Source:
;; void __attribute__((overloadable))
;; foo(image2d_t srcImage);
;;
;; void bar(image2d_t srcImage) {
;;   foo(srcImage);
;; }
;; clang -cc1 /work/tmp/tmp.cl -cl-std=CL2.0 -triple spir-unknown-unknown  -finclude-default-header -emit-llvm -o test/mangled_function.ll

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpName %[[#foo:]] "_Z3foo14ocl_image2d_ro"
; CHECK-SPIRV: %[[#foo]] = OpFunction %[[#]]

%opencl.image2d_ro_t = type opaque

define spir_func void @bar(%opencl.image2d_ro_t addrspace(1)* %srcImage) local_unnamed_addr {
entry:
  tail call spir_func void @_Z3foo14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)* %srcImage)
  ret void
}

declare spir_func void @_Z3foo14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)*) local_unnamed_addr
