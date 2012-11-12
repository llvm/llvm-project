; RUN: llc < %s -march=x86 -mattr=+sse2 | grep lfence

declare void @llvm.x86.sse2.lfence() nounwind

define void @test() {
  call void @llvm.x86.sse2.lfence()
  ret void
}
