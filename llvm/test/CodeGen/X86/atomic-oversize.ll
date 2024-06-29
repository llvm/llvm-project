; RUN: llc -mtriple=x86_64 -mattr=cx16 < %s | FileCheck %s

; Atomics larger than 128-bit are unsupported, and emit libcalls.
define void @test(ptr %a) nounwind {
; CHECK-LABEL: test:
; CHECK: callq __atomic_load
; CHECK: callq __atomic_store
  %1 = load atomic i256, ptr %a seq_cst, align 32
  store atomic i256 %1, ptr %a seq_cst, align 32
  ret void
}
