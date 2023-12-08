; RUN: llc < %s -mtriple=i686-apple-darwin -mattr=+sse2 | FileCheck %s

define void @test() {
  fence acquire
  ; CHECK: #MEMBARRIER

  fence release
  ; CHECK: #MEMBARRIER

  fence acq_rel
  ; CHECK: #MEMBARRIER

  ret void
}
