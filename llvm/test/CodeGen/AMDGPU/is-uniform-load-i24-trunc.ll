; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -o /dev/null < %s

define void @trunc_load_i24(ptr addrspace(4) inreg %in, ptr addrspace(1) inreg %out) {
entry:
  %v = load volatile i24, ptr addrspace(4) %in, align 2
  %t = trunc i24 %v to i16
  store i16 %t, ptr addrspace(1) %out, align 2
  ret void
}
