; RUN: opt -passes=instcombine -S < %s | llvm-as
; RUN: opt -passes='function(instcombine),globalopt' -S < %s | llvm-as
@G1 = global i32 zeroinitializer
@G2 = global i32 zeroinitializer
@g = global <2 x ptr> zeroinitializer
%0 = type { i32, ptr, ptr }
@llvm.global_ctors = appending global [1 x %0] [%0 { i32 65535, ptr @test, ptr null }]
define internal void @test() {
  %A = insertelement <2 x ptr> undef, ptr @G1, i32 0
  %B = insertelement <2 x ptr> %A,  ptr @G2, i32 1
  store <2 x ptr> %B, ptr @g
  ret void
}

