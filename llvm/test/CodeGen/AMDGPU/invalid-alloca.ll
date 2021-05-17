; RUN: not llvm-as -data-layout=A5 < %s 2>&1 | FileCheck -check-prefixes=COMMON,AS %s
; RUN: not llc -mtriple amdgcn-amd-amdhsa < %s 2>&1 | FileCheck -check-prefixes=COMMON,LLC %s
; RUN: llvm-as < %s | not llc -mtriple amdgcn-amd-amdhsa 2>&1 | FileCheck -check-prefixes=MISMATCH %s
; RUN: not opt -data-layout=A5 -S < %s 2>&1 | FileCheck -check-prefixes=COMMON,OPT %s
; RUN: llvm-as < %s | not opt -data-layout=A5 2>&1 | FileCheck -check-prefixes=MISMATCH %s

; AS: assembly parsed, but does not verify as correct!
; COMMON: Allocation instruction pointer not in the stack address space!
; COMMON:  %tmp = alloca i32
; MISMATCH: Explicit load/store type does not match pointee type of pointer operand
; LLC: error: {{.*}}input module cannot be verified
; OPT: error: input module is broken!

define amdgpu_kernel void @test() {
  %tmp = alloca i32
  %tmp2 = alloca i32*
  store i32* %tmp, i32** %tmp2
  ret void
}

