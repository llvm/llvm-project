; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpDecorate %[[#FN:]] LinkageAttributes "" Export
; A spir_kernel is an entry point and must not get a LinkageAttributes decoration.
; CHECK-NOT: OpDecorate %[[#]] LinkageAttributes "" Export
; CHECK: %[[#FN]] = OpFunction

target triple = "spirv64-unknown-unknown"

define void @0(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
entry:
  %ok = icmp sgt i32 %n, 0
  br i1 %ok, label %body, label %exit
body:
  %v = load i32, ptr addrspace(1) %in, align 4
  %salt = mul i32 %n, -1640531527
  %mix = xor i32 %v, %salt
  store i32 %mix, ptr addrspace(1) %out, align 4
  br label %exit
exit:
  ret void
}

define spir_kernel void @1(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %n) {
entry:
  ret void
}
