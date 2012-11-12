; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep sfence
; RUN: llc < %s -march=x86 -mattr=+sse2 | not grep lfence
; RUN: llc < %s -march=x86 -mattr=+sse2 | grep mfence

define void @test() {
  fence seq_cst
  ret void
}
