; RUN: llc -mtriple=bpf -mcpu=generic < %s | FileCheck %s

define void @test(ptr %a) nounwind {
; CHECK-LABEL: test:
; CHECK: r6 = r1
; CHECK-NEXT: r2 = 0
; CHECK-NEXT: call __atomic_load_16
; CHECK-NEXT: r3 = r1
; CHECK-NEXT: r1 = r6
; CHECK-NEXT: r2 = r0
; CHECK-NEXT: r4 = 0
; CHECK-NEXT: call __atomic_store_16
  %1 = load atomic i128, ptr %a monotonic, align 16
  store atomic i128 %1, ptr %a monotonic, align 16
  ret void
}
