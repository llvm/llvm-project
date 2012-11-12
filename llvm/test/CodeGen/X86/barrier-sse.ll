; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep sfence
; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep lfence
; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep mfence
; RUN: llc < %s -march=x86 -mattr=+sse2 | grep MEMBARRIER

define void @test() {
  fence acquire
  fence release
  fence acq_rel
  ret void
}
