; This test shows a few canonicalizations made by deadargelim
; RUN: opt < %s -passes=deadargelim -S > %t
; This test should remove {} and replace it with void
; RUN: cat %t | grep "define internal void @test"
; This test shouls replace the {i32} return value with just i32
; RUN: cat %t | grep "define internal i32 @test2"

define internal {} @test() {
  ret {} undef
}

define internal {i32} @test2() {
  ret {i32} undef
}

define void @caller() {
  call {} @test()
  %X = call {i32} @test2()
  %Y = extractvalue {i32} %X, 0
  call void @user(i32 %Y, {i32} %X)
  ret void
}

declare void @user(i32, {i32})
