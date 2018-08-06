; RUN: llc -mtriple=hexagon-unknown-elf -mcpu=generic < %s | FileCheck %s

; CHECK-NOT: invalid CPU

define i32 @test(i32 %a) {
  ret i32 0
}
