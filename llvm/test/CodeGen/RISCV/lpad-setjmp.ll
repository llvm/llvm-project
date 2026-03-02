; RUN: llc -mtriple riscv32 -mattr=+experimental-zicfilp,+relax,+c \
; RUN:   < %s | FileCheck %s --check-prefixes=CHECK
; RUN: llc -mtriple riscv64 -mattr=+experimental-zicfilp,+relax,+c \
; RUN:   < %s | FileCheck %s --check-prefixes=CHECK
; RUN: llc -mtriple riscv32 -mattr=+experimental-zicfilp,+relax,+zcmt \
; RUN:   < %s | FileCheck %s --check-prefixes=CHECK
; RUN: llc -mtriple riscv32 -mattr=+experimental-zicfilp,-relax,+c \
; RUN:   < %s | FileCheck %s --check-prefixes=NORELAX
; RUN: llc -mtriple riscv32 -mattr=+experimental-zicfilp,+relax,-c \
; RUN:   < %s | FileCheck %s --check-prefixes=NOZCA
; RUN: llc -mtriple riscv64 -mattr=+experimental-zicfilp,+relax,+c \
; RUN:   -riscv-landing-pad-label=1 < %s | FileCheck %s --check-prefixes=LABEL1

; Verify that object file output does not have R_RISCV_RELAX on the setjmp call.
; RUN: llc -mtriple riscv64 -mattr=+experimental-zicfilp,+relax,+c \
; RUN:   -filetype=obj < %s -o %t.o
; RUN: llvm-readobj --relocations %t.o | FileCheck %s --check-prefix=RELOC
; setjmp call: R_RISCV_CALL_PLT followed by regular_func (no R_RISCV_RELAX in between)
; regular_func call: R_RISCV_CALL_PLT followed by R_RISCV_RELAX
; RELOC:      R_RISCV_CALL_PLT setjmp
; RELOC-NEXT: R_RISCV_CALL_PLT regular_func
; RELOC-NEXT: R_RISCV_RELAX

; Test that returns_twice calls (e.g., setjmp) are wrapped with .option push/exact/pop
; when Zicfilp + relax are enabled, to prevent LPAD misalignment.

declare i32 @setjmp(ptr) returns_twice
declare i32 @regular_func(ptr)

define i32 @test_setjmp() {
; CHECK-LABEL: test_setjmp:
; CHECK:         .p2align 2
; CHECK-NEXT:    .option push
; CHECK-NEXT:    .option exact
; CHECK-NEXT:    call setjmp
; CHECK-NEXT:    .option pop
; CHECK-NEXT:    lpad 0
;
; Without relax, only .p2align 2 is needed.
; NORELAX-LABEL: test_setjmp:
; NORELAX-NOT:     .option push
; NORELAX-NOT:     .option exact
; NORELAX:         .p2align 2
; NORELAX-NEXT:    call setjmp
; NORELAX-NEXT:    lpad 0
;
; Without Zca, no .p2align or .option push/exact needed (no compressed calls).
; NOZCA-LABEL: test_setjmp:
; NOZCA-NOT:     .p2align 2
; NOZCA-NOT:     .option push
; NOZCA-NOT:     .option exact
; NOZCA:         call setjmp
; NOZCA-NEXT:    lpad 0
;
; With -riscv-landing-pad-label=1, lpad uses label 1.
; LABEL1-LABEL: test_setjmp:
; LABEL1:         call setjmp
; LABEL1:         lpad 1
  %buf = alloca [1 x i32], align 4
  %call = call i32 @setjmp(ptr %buf)
  ret i32 %call
}

; Regular calls should not be affected.
define i32 @test_regular_call() {
; CHECK-LABEL: test_regular_call:
; CHECK-NOT:     .option exact
; CHECK:         call regular_func
  %buf = alloca [1 x i32], align 4
  %call = call i32 @regular_func(ptr %buf)
  ret i32 %call
}

; Indirect calls with returns_twice attribute also get the same treatment.
define i32 @test_indirect_returns_twice(ptr %fptr) {
; CHECK-LABEL: test_indirect_returns_twice:
; CHECK:         .p2align 2
; CHECK-NEXT:    .option push
; CHECK-NEXT:    .option exact
; CHECK-NEXT:    jalr
; CHECK-NEXT:    .option pop
; CHECK-NEXT:    lpad 0
;
; Without relax, only .p2align 2 is needed (no .option push/exact).
; NORELAX-LABEL: test_indirect_returns_twice:
; NORELAX-NOT:     .option push
; NORELAX-NOT:     .option exact
; NORELAX:         .p2align 2
; NORELAX-NEXT:    jalr
; NORELAX-NEXT:    lpad 0
;
; Without Zca, no .p2align or .option push/exact needed.
; NOZCA-LABEL: test_indirect_returns_twice:
; NOZCA-NOT:     .p2align 2
; NOZCA-NOT:     .option push
; NOZCA-NOT:     .option exact
; NOZCA:         jalr
; NOZCA-NEXT:    lpad 0
;
; With -riscv-landing-pad-label=1, lpad uses label 1.
; LABEL1-LABEL: test_indirect_returns_twice:
; LABEL1:         jalr
; LABEL1:         lpad 1
  %buf = alloca [1 x i32], align 4
  %call = call i32 %fptr(ptr %buf) #0
  ret i32 %call
}

attributes #0 = { returns_twice }

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
