; Note the formula for negative number alignment calculation should be y = x & ~(n-1) rather than y = (x + (n-1)) & ~(n-1).
; after patch https://reviews.llvm.org/D34337, we could save 16 bytes in the best case.
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s -check-prefix=CHECK-BE
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s -check-prefix=CHECK-LE

define signext i32 @bar(i32 signext %ii) {
entry:
  %0 = tail call i32 asm sideeffect "add $0, $1, $2\0A", "=r,r,r,~{f14},~{r15},~{v20}"(i32 %ii, i32 10)
  ret i32 %0
; Before the fix by patch D34337:
; stdu 1, -544(1)
; std 15, 264(1)
; stfd 14, 400(1)
; stdu 1, -560(1)
; std 15, 280(1)
; stfd 14, 416(1)

; After the fix by patch D34337:
; CHECK-LE: stdu 1, -528(1)
; CHECK-LE:std 15, 248(1)
; CHECK-LE:stfd 14, 384(1)
; CHECK-BE: stdu 1, -544(1)
; CHECK-BE:std 15, 264(1)
; CHECK-BE:stfd 14, 400(1)
}

define signext i32 @foo() {
entry:
  %call = tail call signext i32 @bar(i32 signext 5)
  ret i32 %call
}

