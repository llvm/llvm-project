; RUN: llc < %s -mtriple=thumb-apple-darwin | FileCheck %s
; Radar 8143571: Function alignments were off by a power of two.
; CHECK: .p2align 1
define void @test() {
  ret void
}
