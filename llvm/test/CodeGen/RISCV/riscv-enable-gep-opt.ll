; RUN: llc -mtriple=riscv32 -O3 -riscv-enable-gep-opt=true -debug-pass=Structure < %s -o /dev/null 2>&1 | \
; RUN:   grep -v "Verify generated machine code" | \
; RUN:   FileCheck %s --check-prefixes=CHECK


; REQUIRES: asserts

; CHECK-LABEL: Pass Arguments:
; CHECK: Split GEPs to a variadic base and a constant offset for better CSE

