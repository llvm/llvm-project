;; Make sure we emit trap instructions after stack protector checks iff NoTrapAfterNoReturn is false.

; RUN: llc -mtriple=aarch64 -fast-isel=false -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable=false -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=aarch64 -fast-isel=false -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable -no-trap-after-noreturn -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=aarch64 -fast-isel=false -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable -no-trap-after-noreturn=false -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,TRAP_UNREACHABLE %s

;; Make sure FastISel doesn't break anything.
; RUN: llc -mtriple=aarch64 -fast-isel -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable=false -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=aarch64 -fast-isel -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable -no-trap-after-noreturn -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=aarch64 -fast-isel -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable -no-trap-after-noreturn=false -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,TRAP_UNREACHABLE %s

; CHECK-LABEL: Machine code for function test
; CHECK: bb.0.entry:
; CHECK:  BL {{.}}__stack_chk_fail
; CHECK-NEXT: ADJCALLSTACKUP
; TRAP_UNREACHABLE-NEXT: BRK 1
; NO_TRAP_UNREACHABLE-NOT: BRK 1
; NO_TRAP_UNREACHABLE-EMPTY:

define void @test() nounwind ssp {
entry:
  %buf = alloca [8 x i8]
  %result = call i32(ptr) @callee(ptr %buf) nounwind
  ret void
}

declare i32 @callee(ptr) nounwind
