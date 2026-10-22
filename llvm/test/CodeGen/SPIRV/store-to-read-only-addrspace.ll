; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s


; CHECK: error: {{.*}}store into a read-only SPIR-V storage class is not allowed

define spir_kernel void @store_to_constant(ptr addrspace(2) %out) {
  store i32 0, ptr addrspace(2) %out
  ret void
}
