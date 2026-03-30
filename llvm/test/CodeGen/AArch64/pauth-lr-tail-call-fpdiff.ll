; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+v9a,+pauth-lr -o - %s \
; RUN:   | FileCheck %s
;
; When a tail call has FPDiff != 0 and PAuthLR is enabled, the SP-based
; autiasppc instruction uses the wrong SP value (adjusted for the tail
; call's stack args). The fix computes entry SP into x16 and uses
; explicit autia instead.

declare swifttailcc void @callee_stack_args(ptr swiftasync %ctx, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare swifttailcc void @callee_no_stack_args(ptr swiftasync %ctx)

; FPDiff != 0 with PAuthLR: must use explicit autia, not autiasppc.
define swifttailcc void @test_pauthlr_fpdiff(ptr swiftasync %ctx) #0 {
; CHECK-LABEL: test_pauthlr_fpdiff:
; CHECK:         paciasppc
; CHECK-NOT:     autiasppc
; CHECK:         autia x30, x16
; CHECK:         b callee_stack_args
  musttail call swifttailcc void @callee_stack_args(ptr swiftasync %ctx, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9)
  ret void
}

; FPDiff == 0 with PAuthLR: autiasppc is correct.
define swifttailcc void @test_pauthlr_no_fpdiff(ptr swiftasync %ctx) #0 {
; CHECK-LABEL: test_pauthlr_no_fpdiff:
; CHECK:         paciasppc
; CHECK:         autiasppc
; CHECK-NOT:     autia x30, x16
; CHECK:         b callee_no_stack_args
  musttail call swifttailcc void @callee_no_stack_args(ptr swiftasync %ctx)
  ret void
}

attributes #0 = { nounwind "branch-protection-pauth-lr" "sign-return-address"="all" "frame-pointer"="all" }
