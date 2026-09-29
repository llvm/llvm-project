; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; The custom-ABI preserve counts must parse as unsigned base-ten integers.

; CHECK: "nvvm.preserve_n_data" takes an unsigned integer: foobar
define void @not_an_integer() "nvvm.preserve_n_data"="foobar" {
  ret void
}

; CHECK: "nvvm.preserve_n_control" takes an unsigned integer: -1
define void @negative() "nvvm.preserve_n_control"="-1" {
  ret void
}

; CHECK: "nvvm.preserve_n_data" takes an unsigned integer: 8,2
define void @not_a_vector() "nvvm.preserve_n_data"="8,2" {
  ret void
}

; The same check applies at call sites, which is where an indirect call carries
; its contract. verifyFunctionAttrs() is reached from visitCallBase() as well as
; from the function walk.
; CHECK: "nvvm.preserve_n_control" takes an unsigned integer: bad
define void @bad_call_site(ptr %fp) {
  call void %fp() "nvvm.preserve_n_control"="bad"
  ret void
}

; PTX only permits the directives between a .func directive and its body, so a
; kernel cannot express them at all.
; CHECK: 'nvvm.preserve_n_data' is not allowed on kernel functions
define ptx_kernel void @kernel_data() "nvvm.preserve_n_data"="8" {
  ret void
}

; CHECK: 'nvvm.preserve_n_control' is not allowed on kernel functions
define ptx_kernel void @kernel_control() "nvvm.preserve_n_control"="2" {
  ret void
}

; A device function may carry them; only kernels are rejected.
define void @device_fn_is_fine() "nvvm.preserve_n_data"="8" "nvvm.preserve_n_control"="2" {
  ret void
}
