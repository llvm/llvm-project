; RUN: not llvm-as -data-layout=A5 < %s 2>&1 | FileCheck -check-prefixes=COMMON,AS %s
; RUN: not llc -mtriple amdgcn-amd-amdhsa < %s 2>&1 | FileCheck -check-prefixes=COMMON,LLC %s
; RUN: llvm-as < %s | not llc -mtriple amdgcn-amd-amdhsa 2>&1 | FileCheck -check-prefixes=COMMON,LLC %s
; RUN: not opt -data-layout=A5 -S < %s 2>&1 | FileCheck -check-prefixes=COMMON,LLC %s
; RUN: llvm-as < %s | not opt -data-layout=A5 2>&1 | FileCheck -check-prefixes=COMMON,LLC %s

; AS: assembly parsed, but does not verify as correct!
; COMMON: Allocation instruction pointer not in the stack address space!
; COMMON:  %tmp = alloca i32
; LLC: error: input module is broken!

define amdgpu_kernel void @test() {
  %tmp = alloca i32
  ret void
}

