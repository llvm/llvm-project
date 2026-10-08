; Test that RISCVISelDAGToDAG correctly selects PseudoCALLLpadAlign and
; PseudoCALLIndirectLpadAlign for RISCVISD::LPAD_CALL and
; RISCVISD::LPAD_CALL_INDIRECT nodes (returns_twice calls with CFI enabled).
; LPAD pseudo selection is triggered by the cf-protection-branch module flag,
; not by +experimental-zicfilp.

; RUN: llc -mtriple=riscv32 -stop-after=finalize-isel \
; RUN:   < %s | FileCheck %s --check-prefixes=CFP
; RUN: llc -mtriple=riscv64 -stop-after=finalize-isel \
; RUN:   < %s | FileCheck %s --check-prefixes=CFP
; RUN: llc -mtriple=riscv64 -riscv-landing-pad-label=1 \
; RUN:   -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=LABEL1

declare i32 @setjmp(ptr) returns_twice

; Direct returns_twice call: should select PseudoCALLLpadAlign.
define i32 @test_direct() {
  ; CFP-LABEL: name: test_direct
  ; CFP: PseudoCALLLpadAlign target-flags(riscv-call) @setjmp, 0
  ; LABEL1-LABEL: name: test_direct
  ; LABEL1: PseudoCALLLpadAlign target-flags(riscv-call) @setjmp, 1
  %buf = alloca [1 x i32], align 4
  %call = call i32 @setjmp(ptr %buf)
  ret i32 %call
}

; Indirect returns_twice call: should select PseudoCALLIndirectLpadAlign.
define i32 @test_indirect(ptr %fptr) {
  ; CFP-LABEL: name: test_indirect
  ; CFP: PseudoCALLIndirectLpadAlign %{{[0-9]+}}, 0
  ; LABEL1-LABEL: name: test_indirect
  ; LABEL1: PseudoCALLIndirectLpadAlign %{{[0-9]+}}, 1
  %call = call i32 %fptr() #0
  ret i32 %call
}

; Non-returns_twice calls must NOT select PseudoCALLLpadAlign.
declare i32 @regular_func()

define i32 @test_regular_call() {
  ; CFP-LABEL: name: test_regular_call
  ; CFP-NOT: PseudoCALLLpadAlign
  ; CFP: PseudoCALL target-flags(riscv-call) @regular_func
  %call = call i32 @regular_func()
  ret i32 %call
}

attributes #0 = { returns_twice }

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
