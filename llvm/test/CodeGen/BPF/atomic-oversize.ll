; RUN: llc -mtriple=bpf < %s | FileCheck %s

define void @test(ptr %a) nounwind {
  %1 = load atomic i128, ptr %a monotonic, align 16
  store atomic i128 %1, ptr %a monotonic, align 16
  ret void
}
