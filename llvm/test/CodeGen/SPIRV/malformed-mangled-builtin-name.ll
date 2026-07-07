; A call target with a malformed mangled name must not crash the demangler.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null

declare spir_func i64 @"_Z&3gmt_global_idj"(i32)

define spir_kernel void @fuzz_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
entry:
  %id = call spir_func i64 @"_Z&3gmt_global_idj"(i32 0)
  %idx = trunc i64 %id to i32
  %gep = getelementptr i32, ptr addrspace(1) %in, i32 %idx
  %v = load i32, ptr addrspace(1) %gep, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
