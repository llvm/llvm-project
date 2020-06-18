; RUN: opt -mtriple=amdgcn-- -O1 -S < %s | FileCheck %s --check-prefixes=FUNC,LOOP
; RUN: opt -mtriple=amdgcn-- -O1 -S -disable-promote-alloca-to-vector < %s | FileCheck %s --check-prefixes=FUNC,FULL-UNROLL

target datalayout = "A5"

; This test contains a simple loop that initializes an array declared in
; private memory. This loop would be fully unrolled if we could not SROA
; the alloca. Check that we successfully eliminate it before the unroll,
; so that we do not need to fully unroll it.

; FUNC-LABEL: @private_memory
; LOOP-NOT: alloca
; LOOP: loop.header:
; LOOP: br i1 %{{[^,]+}}, label %exit, label %loop.header

; FULL-UNROLL: alloca
; FULL-UNROLL-COUNT-256: store i32 {{[0-9]+}}, i32 addrspace(5)*
; FULL-UNROLL-NOT: br

; FUNC: store i32 %{{[^,]+}}, i32 addrspace(1)* %out
define amdgpu_kernel void @private_memory(i32 addrspace(1)* %out, i32 %n) {
entry:
  %alloca = alloca [16 x i32], addrspace(5)
  br label %loop.header

loop.header:
  %counter = phi i32 [0, %entry], [%inc, %loop.inc]
  br label %loop.body

loop.body:
  %salt = xor i32 %counter, %n
  %idx = and i32 %salt, 15
  %ptr = getelementptr [16 x i32], [16 x i32] addrspace(5)* %alloca, i32 0, i32 %idx
  store i32 %counter, i32 addrspace(5)* %ptr
  br label %loop.inc

loop.inc:
  %inc = add i32 %counter, 1
  %cmp = icmp sge i32 %counter, 255
  br i1 %cmp, label  %exit, label %loop.header

exit:
  %gep = getelementptr [16 x i32], [16 x i32] addrspace(5)* %alloca, i32 0, i32 %n
  %load = load i32, i32 addrspace(5)* %gep
  store i32 %load, i32 addrspace(1)* %out
  ret void
}
