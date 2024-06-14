; RUN: llc -fast-isel -mtriple=i686-- < %s | FileCheck %s
; CHECK: %fs:

define i32 @test1(ptr addrspace(257) %arg) nounwind {
       %tmp = load i32, ptr addrspace(257) %arg
       ret i32 %tmp
}
