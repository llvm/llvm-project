; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
target datalayout = "p1:64:64:64:32"

@i_as0 = global i32 0
@global_cast_as0 = global i64 ptrtoaddr (ptr @i_as0 to i64)
; CHECK: @global_cast_as0 = global i64 ptrtoaddr (ptr @i_as0 to i64)
@i_as1 = addrspace(1) global i32 0
@global_cast_as1 = global i32 ptrtoaddr (ptr addrspace(1) @i_as1 to i32)
; CHECK: @global_cast_as1 = global i32 ptrtoaddr (ptr addrspace(1) @i_as1 to i32)

define i64 @test_as0(ptr %p) {
  %addr = ptrtoaddr ptr %p to i64
  ; CHECK: %addr = ptrtoaddr ptr %p to i64
  ret i64 %addr
}

define i32 @test_as1(ptr addrspace(1) %p) {
  %addr = ptrtoaddr ptr addrspace(1) %p to i32
  ; CHECK: %addr = ptrtoaddr ptr addrspace(1) %p to i32
  ret i32 %addr
}

define <2 x i32> @test_vec_as1(<2 x ptr addrspace(1)> %p) {
  %addr = ptrtoaddr <2 x ptr addrspace(1)> %p to <2 x i32>
  ; CHECK: %addr = ptrtoaddr <2 x ptr addrspace(1)> %p to <2 x i32>
  ret <2 x i32> %addr
}
