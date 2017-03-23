; RUN: opt -mtriple=amdgcn-- -loop-unroll -simplifycfg -sroa %s -S -o - | FileCheck %s
; RUN: opt -mtriple=r600-- -loop-unroll -simplifycfg -sroa %s -S -o - | FileCheck %s


; This test contains a simple loop that initializes an array declared in
; private memory.  We want to make sure these kinds of loops are always
; unrolled, because private memory is slow.

; CHECK-LABEL: @test
; CHECK-NOT: alloca
; CHECK: store i32 5, i32 addrspace(1)* %out
define amdgpu_kernel void @test(i32 addrspace(1)* %out) {
entry:
  %0 = alloca [32 x i32]
  br label %loop.header

loop.header:
  %counter = phi i32 [0, %entry], [%inc, %loop.inc]
  br label %loop.body

loop.body:
  %ptr = getelementptr [32 x i32], [32 x i32]* %0, i32 0, i32 %counter
  store i32 %counter, i32* %ptr
  br label %loop.inc

loop.inc:
  %inc = add i32 %counter, 1
  %1 = icmp sge i32 %counter, 32
  br i1 %1, label  %exit, label %loop.header

exit:
  %2 = getelementptr [32 x i32], [32 x i32]* %0, i32 0, i32 5
  %3 = load i32, i32* %2
  store i32 %3, i32 addrspace(1)* %out
  ret void
}
