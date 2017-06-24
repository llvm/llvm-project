; RUN: llc < %s -O3 -debug-pass=Structure -print-machineinstrs=branch-folder -verify-machineinstrs -o /dev/null 2>&1 \
; RUN:   | FileCheck %s -check-prefix=PRINT-BRANCH-FOLD
; RUN: llc < %s -O3 -debug-pass=Structure -print-machineinstrs -verify-machineinstrs -o /dev/null 2>&1 \
; RUN:   | FileCheck %s -check-prefix=PRINT
; RUN: llc < %s -O3 -debug-pass=Structure -print-machineinstrs= -verify-machineinstrs -o /dev/null 2>&1 \
; RUN:   | FileCheck %s -check-prefix=PRINT

; Note: -verify-machineinstrs is used in order to make this test compatible with EXPENSIVE_CHECKS.

define i64 @foo(i64 %a, i64 %b) nounwind {
; PRINT-BRANCH-FOLD: -branch-folder -machineverifier -machineinstr-printer
; PRINT-BRANCH-FOLD: Control Flow Optimizer
; PRINT-BRANCH-FOLD-NEXT: Verify generated machine code
; PRINT-BRANCH-FOLD-NEXT: MachineFunction Printer
; PRINT-BRANCH-FOLD: Machine code for function foo:

; PRINT: -branch-folder -machineinstr-printer
; PRINT: Control Flow Optimizer
; PRINT-NEXT: MachineFunction Printer
; PRINT-NEXT: Verify generated machine code
; PRINT: Machine code for function foo:

  %c = add i64 %a, %b
  %d = trunc i64 %c to i32
  %e = zext i32 %d to i64
  ret i64 %e
}
