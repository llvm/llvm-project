@.str = external addrspace(1) constant [21 x i8]

define spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_() {
entry:
  %call.i = tail call spir_func i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4)), ptr addrspace(4) null, ptr addrspace(4) null, i32 0, i32 0, i32 0, ptr addrspace(4) null)
  ret void
}

declare spir_func i32 @printf(ptr addrspace(4), ...)
