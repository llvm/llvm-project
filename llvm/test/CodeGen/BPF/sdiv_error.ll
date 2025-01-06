; RUN: not llc -mtriple=bpf < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: unsupported signed division

define i64 @test(i64 %len) {
  %1 = sdiv i64 %len, 15
  ret i64 %1
}
