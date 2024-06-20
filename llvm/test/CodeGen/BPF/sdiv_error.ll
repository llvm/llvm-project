; RUN: not llc -mtriple=bpf < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: unsupported signed division

define i32 @test(i32 %len) {
  %1 = sdiv i32 %len, 15
  ret i32 %1
}
