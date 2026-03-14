; RUN: opt -passes=lint < %s

target datalayout = "p32:32:32-p1:16:16:16-n16:32"

declare void @foo(i64) nounwind

define i64 @test1(ptr addrspace(1) %x) nounwind {
  %y = ptrtoint ptr addrspace(1) %x to i64
  ret i64 %y
}

define <4 x i64> @test1_vector(<4 x ptr addrspace(1)> %x) nounwind {
  %y = ptrtoint <4 x ptr addrspace(1)> %x to <4 x i64>
  ret <4 x i64> %y
}

define ptr addrspace(1) @test2(i64 %x) nounwind {
  %y = inttoptr i64 %x to ptr addrspace(1)
  ret ptr addrspace(1) %y
}

define <4 x ptr addrspace(1)> @test2_vector(<4 x i64> %x) nounwind {
  %y = inttoptr <4 x i64> %x to <4 x ptr addrspace(1)>
  ret <4 x ptr addrspace(1)> %y
}