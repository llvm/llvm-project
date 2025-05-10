; RUN: llvm-as < %s | llvm-dis | FileCheck %s

target datalayout = "p1:64:64:64:32"

define i32 @test_as0(ptr %p) {
  %addr = ptrtoaddr ptr %p to i32
  ; CHECK: %addr = ptrtoaddr ptr %p to i32
  ret i32 %addr
}

define i16 @test_as1(ptr addrspace(1) %p) {
  %addr = ptrtoaddr ptr addrspace(1) %p to i16
  ; CHECK: %addr = ptrtoaddr ptr addrspace(1) %p to i16
  ret i16 %addr
}
