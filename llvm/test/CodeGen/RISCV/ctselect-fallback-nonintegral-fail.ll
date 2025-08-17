; XFAIL: *
; RUN: llc -mtriple=riscv32 < %s | FileCheck %s 
; RUN: llc -mtriple=riscv64 < %s | FileCheck %s 

; This test should fail
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128-ni:200"
target triple = "riscv64-unknown-linux-gnu"

define i32 addrspace(200)* @test_ctselect_ptr(i1 %c, 
                                              i32 addrspace(200)* %a, 
                                              i32 addrspace(200)* %b) {
  %r = call i32 addrspace(200)* @llvm.ct.select.p0(i1 %c, 
                                                     i32 addrspace(200)* %a, 
                                                     i32 addrspace(200)* %b)
  ret i32 addrspace(200)* %r
}

declare i32 @llvm.ct.select.p0(i1, i32, i32)
