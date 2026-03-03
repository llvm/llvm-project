; RUN: llc -mtriple arm64e-apple-darwin -o - %s | FileCheck %s
;
; Crash 13 repro: In Swift async functions using swifttailcc, a tail call
; with stack arguments adjusts SP in the epilogue before return address
; authentication. AUTIBSP uses the current (adjusted) SP, not the entry
; SP from PACIBSP, causing EXC_ARM_PAC_FAIL on arm64e.
;
; Fix: When FPDiff != 0, compute the entry SP into x16 and use explicit
; autib x30, x16 instead of autibsp.

declare swifttailcc void @callee_async(ptr swiftasync %ctx, i64, i64, i64, i64, i64, i64, i64, i64, i64)

; FPDiff != 0: callee has stack args that this function doesn't.
; Must use explicit autib with computed entry SP, NOT autibsp.
define swifttailcc void @test_async_tail_call(ptr swiftasync %ctx) #0 {
; CHECK-LABEL: _test_async_tail_call:
; CHECK:         pacibsp
; CHECK-NOT:     autibsp
; CHECK:         autib x30, x16
; CHECK:         b _callee_async
  musttail call swifttailcc void @callee_async(ptr swiftasync %ctx, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9)
  ret void
}

declare swifttailcc void @callee_no_stack_args(ptr swiftasync %ctx)

; FPDiff == 0: callee has same stack arg layout. autibsp is correct here.
define swifttailcc void @test_no_fpdiff_tail_call(ptr swiftasync %ctx) #0 {
; CHECK-LABEL: _test_no_fpdiff_tail_call:
; CHECK:         pacibsp
; CHECK:         autibsp
; CHECK-NOT:     autib x30, x16
; CHECK:         b _callee_no_stack_args
  musttail call swifttailcc void @callee_no_stack_args(ptr swiftasync %ctx)
  ret void
}

attributes #0 = { nounwind "ptrauth-returns" "ptrauth-auth-traps" "sign-return-address"="all" "frame-pointer"="all" }
