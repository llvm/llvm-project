; RUN: llc -debug-only=machine-scheduler -march=amdgcn -mtriple=amdgcn-- -verify-machineinstrs %s -o - 2>&1| FileCheck -check-prefix=DEBUG %s
target datalayout = "A5"
; REQUIRES: asserts

; Verify that the extload generated from %eval has the default
; alignment size (2) corresponding to the underlying memory size (i16)
; size and not 4 corresponding to the sign-extended size (i32).

; DEBUG: {{^}}# Machine code for function extload_align:
; DEBUG: (volatile load (s16) from %ir.a, addrspace 5)
; DEBUG: {{^}}# End machine code for function extload_align.

define amdgpu_kernel void @extload_align(ptr addrspace(5) %out, i32 %index) #0 {
  %v0 = alloca [4 x i16], addrspace(5)
  %a2 = getelementptr inbounds [4 x i16], ptr addrspace(5) %v0, i32 0, i32 1
  store volatile i16 0, ptr addrspace(5) %v0
  store volatile i16 1, ptr addrspace(5) %a2
  %a = getelementptr inbounds [4 x i16], ptr addrspace(5) %v0, i32 0, i32 %index
  %val = load volatile i16, ptr addrspace(5) %a
  %eval = sext i16 %val to i32
  store i32 %eval, ptr addrspace(5) %out
  ret void
}
