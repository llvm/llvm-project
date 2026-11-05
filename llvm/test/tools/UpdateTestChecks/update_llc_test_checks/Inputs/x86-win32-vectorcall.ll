
; RUN: llc -mtriple=i686-pc-win32 -mattr=+sse2 < %s | FileCheck %s

define x86_vectorcallcc i32 @test_int_1() {
  ret i32 0
}

define x86_vectorcallcc i32 @test_int_2(i32 %a) {
  ret i32 %a
}
