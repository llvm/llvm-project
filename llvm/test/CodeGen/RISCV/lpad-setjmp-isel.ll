; Test that RISCVISelDAGToDAG correctly selects PseudoCALLLpadAlign and
; PseudoCALLIndirectLpadAlign for RISCVISD::LPAD_CALL and
; RISCVISD::LPAD_CALL_INDIRECT nodes (returns_twice calls with Zicfilp + CFI).

; RUN: llc -mtriple=riscv32 -mattr=+experimental-zicfilp -stop-after=finalize-isel \
; RUN:   < %s | FileCheck %s --check-prefixes=ZICFILP
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zicfilp -stop-after=finalize-isel \
; RUN:   < %s | FileCheck %s --check-prefixes=ZICFILP
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zicfilp -riscv-landing-pad-label=1 \
; RUN:   -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=LABEL1
; RUN: llc -mtriple=riscv64 -stop-after=finalize-isel \
; RUN:   < %s | FileCheck %s --check-prefixes=NOZICFILP

declare i32 @setjmp(ptr) returns_twice

; Direct returns_twice call: should select PseudoCALLLpadAlign.
define i32 @test_direct() {
  ; ZICFILP-LABEL: name: test_direct
  ; ZICFILP: PseudoCALLLpadAlign target-flags(riscv-call) @setjmp, 0
  ; LABEL1-LABEL: name: test_direct
  ; LABEL1: PseudoCALLLpadAlign target-flags(riscv-call) @setjmp, 1
  ; Without Zicfilp, a regular PseudoCALL is used instead.
  ; NOZICFILP-LABEL: name: test_direct
  ; NOZICFILP-NOT: PseudoCALLLpadAlign
  ; NOZICFILP: PseudoCALL target-flags(riscv-call) @setjmp
  %buf = alloca [1 x i32], align 4
  %call = call i32 @setjmp(ptr %buf)
  ret i32 %call
}

; Indirect returns_twice call: should select PseudoCALLIndirectLpadAlign.
define i32 @test_indirect(ptr %fptr) {
  ; ZICFILP-LABEL: name: test_indirect
  ; ZICFILP: PseudoCALLIndirectLpadAlign %{{[0-9]+}}, 0
  ; LABEL1-LABEL: name: test_indirect
  ; LABEL1: PseudoCALLIndirectLpadAlign %{{[0-9]+}}, 1
  ; Without Zicfilp, a regular PseudoCALLIndirect is used.
  ; NOZICFILP-LABEL: name: test_indirect
  ; NOZICFILP-NOT: PseudoCALLIndirectLpadAlign
  ; NOZICFILP: PseudoCALLIndirect %{{[0-9]+}}
  %call = call i32 %fptr() #0
  ret i32 %call
}

; Non-returns_twice calls must NOT select PseudoCALLLpadAlign.
declare i32 @regular_func()

define i32 @test_regular_call() {
  ; ZICFILP-LABEL: name: test_regular_call
  ; ZICFILP-NOT: PseudoCALLLpadAlign
  ; ZICFILP: PseudoCALL target-flags(riscv-call) @regular_func
  %call = call i32 @regular_func()
  ret i32 %call
}

attributes #0 = { returns_twice }

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
