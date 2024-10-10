; NOTE: Do not autogenerate, we'd lose the .Lfunc_end0 -NEXT checks otherwise.
;; Make sure we emit trap instructions after stack protector checks iff NoTrapAfterNoReturn is false.

; RUN: llc -mtriple=aarch64 -fast-isel=false -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable=false -o - %s | FileCheck --check-prefix=NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=aarch64 -fast-isel=false -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn -o - %s | FileCheck --check-prefix=NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=aarch64 -fast-isel=false -global-isel=false -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn=false -o - %s | FileCheck --check-prefix=TRAP_UNREACHABLE %s

;; Make sure FastISel doesn't break anything.
; RUN: llc -mtriple=aarch64 -fast-isel -verify-machineinstrs \
; RUN:     -trap-unreachable=false -o - %s | FileCheck --check-prefix=NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=aarch64 -fast-isel -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn -o - %s | FileCheck --check-prefix=NO_TRAP_UNREACHABLE %s
; RUN: llc -mtriple=aarch64 -fast-isel -verify-machineinstrs \
; RUN:     -trap-unreachable -no-trap-after-noreturn=false -o - %s | FileCheck --check-prefix=TRAP_UNREACHABLE %s

define void @test() nounwind ssp {
; NO_TRAP_UNREACHABLE-LABEL: test:
; NO_TRAP_UNREACHABLE:         bl __stack_chk_fail
; NO_TRAP_UNREACHABLE-NOT:     brk #0x1
; NO_TRAP_UNREACHABLE-NEXT:  .Lfunc_end0
;
; TRAP_UNREACHABLE-LABEL: test:
; TRAP_UNREACHABLE:         bl __stack_chk_fail
; TRAP_UNREACHABLE-NEXT:    brk #0x1
; TRAP_UNREACHABLE-NEXT:  .Lfunc_end0

entry:
  %buf = alloca [8 x i8]
  %2 = call i32(ptr) @callee(ptr %buf) nounwind
  ret void
}

declare i32 @callee(ptr) nounwind
