; RUN: llc -fast-isel -march=x86 < %s | grep %fs:

define i32 @test1(i32 addrspace(257)* %arg) nounwind {
       %tmp = load i32 addrspace(257)* %arg
       ret i32 %tmp
}
