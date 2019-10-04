; RUN: llc -mtriple=thumbv7k-apple-watchos %s -o - | FileCheck %s

; r7 is FP on Darwin, and should be preserved even if we don't create a new
; frame record for this leaf function. So make huge register pressure to try &
; tempt LLVM to use it.
define void @foo([16 x i32]* %ptr) {
; CHECK-LABEL: foo:
; CHECK: push.w
; CHECK: .cfi_offset r7
; CHECK-NOT: r7
; CHECK: pop.w
  %val = load volatile [16 x i32], [16 x i32]* %ptr
  store volatile [16 x i32] %val, [16 x i32]* %ptr
  ret void
}
