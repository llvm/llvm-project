; RUN: llc -mtriple=msp430 < %s | FileCheck %s

; Native atomics are unsupported, so all are oversize.
define void @test(ptr %a) nounwind {
; CHECK-LABEL: test:
; CHECK: call #__atomic_load_1
; CHECK: call #__atomic_store_1
  %1 = load atomic i8, ptr %a monotonic, align 16
  store atomic i8 %1, ptr %a monotonic, align 16
  ret void
}
