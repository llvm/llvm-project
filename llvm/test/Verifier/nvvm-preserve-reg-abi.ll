; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: "nvvm.abi_preserve" takes an unsigned integer: foobar
define void @not_an_integer() "nvvm.abi_preserve"="foobar" {
  ret void
}

; CHECK: "nvvm.abi_preserve_control" takes an unsigned integer: -1
define void @negative() "nvvm.abi_preserve_control"="-1" {
  ret void
}

; CHECK: "nvvm.abi_preserve" takes an unsigned integer: 8,2
define void @not_a_vector() "nvvm.abi_preserve"="8,2" {
  ret void
}

; CHECK: "nvvm.abi_preserve_control" takes an unsigned integer: bad
define void @bad_call_site(ptr %fp) {
  call void %fp() "nvvm.abi_preserve_control"="bad"
  ret void
}

; CHECK: 'nvvm.abi_preserve' is not allowed on kernel functions
define ptx_kernel void @kernel_data() "nvvm.abi_preserve"="8" {
  ret void
}

; CHECK: 'nvvm.abi_preserve_control' is not allowed on kernel functions
define ptx_kernel void @kernel_control() "nvvm.abi_preserve_control"="2" {
  ret void
}

define void @device_fn_is_fine() "nvvm.abi_preserve"="8" "nvvm.abi_preserve_control"="2" {
  ret void
}
