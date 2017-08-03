; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

; CHECK-LABEL: test
; CHECK: vpxord
; CHECK: ret
define <16 x i32> @test() {
entry:
  %0 = icmp slt <16 x i32> undef, undef
  %1 = select <16 x i1> %0, <16 x i32> undef, <16 x i32> zeroinitializer
  ret <16 x i32> %1
}
