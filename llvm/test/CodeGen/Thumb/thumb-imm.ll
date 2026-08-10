; RUN: llc -mtriple=thumb-eabi %s -o - | FileCheck %s

define i32 @test1() {
  ret i32 1000
}

define i32 @test2() {
  ret i32 -256
}

; CHECK-NOT: CPI

