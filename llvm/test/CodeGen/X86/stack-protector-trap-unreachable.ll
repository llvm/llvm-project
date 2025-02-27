;; Make sure we emit trap instructions after stack protector checks iff NoTrapAfterNoReturn is false.

; RUN: llc -enable-selectiondag-sp -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable=false -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,NO_TRAP_UNREACHABLE %s
; RUN: llc -enable-selectiondag-sp -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable -no-trap-after-noreturn -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,NO_TRAP_UNREACHABLE %s
; RUN: llc -enable-selectiondag-sp -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable -no-trap-after-noreturn=false -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,TRAP_UNREACHABLE %s

; RUN: llc -enable-selectiondag-sp=false -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable=false -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,NO_TRAP_UNREACHABLE %s
; RUN: llc -enable-selectiondag-sp=false -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable -no-trap-after-noreturn -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,NO_TRAP_UNREACHABLE %s
; RUN: llc -enable-selectiondag-sp=false -mtriple=x86_64 -fast-isel=false -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable -no-trap-after-noreturn=false -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,TRAP_UNREACHABLE %s

;; Make sure FastISel doesn't break anything.
; RUN: llc -mtriple=x86_64 -fast-isel -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable=false --o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=x86_64 -fast-isel -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable -no-trap-after-noreturn -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=x86_64 -fast-isel -global-isel=false -verify-machineinstrs -print-after=finalize-isel \
; RUN:     -trap-unreachable -no-trap-after-noreturn=false -o /dev/null 2>&1 %s | FileCheck --check-prefixes=CHECK,TRAP_UNREACHABLE %s

; CHECK-LABEL: Machine code for function test
; CHECK: bb.0.entry:
; CHECK: CALL64{{.*}}__stack_chk_fail
; CHECK-NEXT: ADJCALLSTACKUP64
; TRAP_UNREACHABLE-NEXT: TRAP
; NO_TRAP_UNREACHABLE-NOT: TRAP
; NO_TRAP_UNREACHABLE-EMPTY:

define void @test() nounwind ssp {
entry:
  %buf = alloca [8 x i8]
  %result = call i32(ptr) @callee(ptr %buf) nounwind
  ret void
}

declare i32 @callee(ptr) nounwind
