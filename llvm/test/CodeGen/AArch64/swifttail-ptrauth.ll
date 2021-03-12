; RUN: llc -verify-machineinstrs < %s -mtriple=arm64e-apple-macosx | FileCheck %s

declare swifttailcc void @callee_stack0()
declare swifttailcc void @callee_stack8([8 x i64], i64)
declare swifttailcc void @callee_stack16([8 x i64], i64, i64)
declare extern_weak swifttailcc void @callee_weak()

define swifttailcc void @caller_to0_from0() "ptrauth-returns" "frame-pointer"="all" nounwind {
; CHECK-LABEL: caller_to0_from0:
; CHECK: stp x29, x30, [sp, #-16]!
; [...]
; CHECK: ldp x29, x30, [sp], #16
; CHECK-NEXT: autibsp
; CHECK-NOT: add sp
; CHECK-NOT: sub sp
  musttail call swifttailcc void @callee_stack0()
  ret void

}

define swifttailcc void @caller_to0_from8([8 x i64], i64) "ptrauth-returns" "frame-pointer"="all" {
; CHECK-LABEL: caller_to0_from8:
; CHECK: stp x29, x30, [sp, #-16]!
; [...]
; CHECK: ldp x29, x30, [sp], #16
; CHECK-NEXT: autibsp
; CHECK: add sp, sp, #16

  musttail call swifttailcc void @callee_stack0()
  ret void

}

define swifttailcc void @caller_to8_from0() "ptrauth-returns" "frame-pointer"="all" {
; CHECK-LABEL: caller_to8_from0:
; CHECK: stp x29, x30, [sp, #-32]!
; [...]
; CHECK: ldp x29, x30, [sp], #16
; CHECK-NEXT: add x16, sp, #16
; CHECK-NEXT: autib x30, x16
; CHECK-NOT: add sp
; CHECK-NOT: sub sp

; Key point is that we don't move sp then autibsp because that leaves live
; arguments below sp, potentially outside the redzone.
  musttail call swifttailcc void @callee_stack8([8 x i64] undef, i64 42)
  ret void

}
