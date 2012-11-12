; RUN: llc < %s -march=x86 -mattr=+sse2 | grep sfence

declare void @llvm.x86.sse.sfence() nounwind

define void @test() {
  call void @llvm.x86.sse.sfence()
  ret void
}
