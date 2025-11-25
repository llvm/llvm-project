; RUN: llc -mtriple=bpf < %s | FileCheck %s

define void @test(ptr %a) nounwind {
; CHECK-LABEL: test:
; CHECK: call __atomic_load_16
; CHECK: call __atomic_store_16
  %1 = load atomic i128, ptr %a monotonic, align 16
  store atomic i128 %1, ptr %a monotonic, align 16
  ret void
}
