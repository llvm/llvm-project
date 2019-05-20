; RUN: opt -instcombine -S < %s | llvm-as
; RUN: opt -instcombine -globalopt -S < %s | llvm-as
@G1 = global i32 zeroinitializer
@G2 = global i32 zeroinitializer
@g = global <2 x i32*> zeroinitializer
%0 = type { i32, void ()* }
@llvm.global_ctors = appending global [1 x %0] [%0 { i32 65535, void ()* @test }]
define internal void @test() {
  %A = insertelement <2 x i32*> undef, i32* @G1, i32 0
  %B = insertelement <2 x i32*> %A,  i32* @G2, i32 1
  store <2 x i32*> %B, <2 x i32*>* @g
  ret void
}

