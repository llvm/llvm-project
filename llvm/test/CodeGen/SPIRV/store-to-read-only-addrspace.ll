; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; Storing into addrspace(2) (__constant / UniformConstant) is read-only in SPIR-V.
; The backend must diagnose this rather than emit invalid SPIR-V.

; CHECK: error: {{.*}}store into a read-only SPIR-V storage class is not allowed

define spir_kernel void @fuzz_kernel(ptr addrspace(1) %in, ptr addrspace(2) %out, i32 %n) {
entry:
  %ok = icmp sgt i32 %n, 0
  br i1 %ok, label %body, label %exit
body:
  %v = load i32, ptr addrspace(1) %in, align 4
  %salt = mul i32 %n, -1640531527
  %mix = xor i32 %v, %salt
  store i32 %mix, ptr addrspace(2) %out, align 4
  br label %exit
exit:
  ret void
}
