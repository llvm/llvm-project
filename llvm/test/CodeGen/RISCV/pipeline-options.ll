; RUN: llc -mtriple=riscv64 -O3  \
; RUN: -debug-pass=Structure < %s -o /dev/null 2>&1 | \
; RUN:   FileCheck %s --check-prefix=O3-WITHOUT-ENABLE-CFI-INSTR-INSERTER

; RUN: llc -mtriple=riscv64 -O3  \
; RUN: --riscv-enable-cfi-instr-inserter=true \
; RUN: -debug-pass=Structure < %s -o /dev/null 2>&1 | \
; RUN:   FileCheck %s --check-prefix=O3-ENABLE-CFI-INSTR-INSERTER

; RUN: llc -mtriple=riscv64 -O0  \
; RUN: -debug-pass=Structure < %s -o /dev/null 2>&1 | \
; RUN:   FileCheck %s --check-prefix=O0-WITHOUT-ENABLE-CFI-INSTR-INSERTER

; RUN: llc -mtriple=riscv64 -O0  \
; RUN: --riscv-enable-cfi-instr-inserter=true \
; RUN: -debug-pass=Structure < %s -o /dev/null 2>&1 | \
; RUN:   FileCheck %s --check-prefix=O0-ENABLE-CFI-INSTR-INSERTER

; REQUIRES: asserts

; O3-WITHOUT-ENABLE-CFI-INSTR-INSERTER-LABEL: Pass Arguments:
; NO-O3-WITHOUT-ENABLE-CFI-INSTR-INSERTER: Check CFA info and insert CFI instructions if needed
; O3-WITHOUT-ENABLE-CFI-INSTR-INSERTER: Insert CFI remember/restore state instructions

; O3-ENABLE-CFI-INSTR-INSERTER-LABEL: Pass Arguments:
; O3-ENABLE-CFI-INSTR-INSERTER: Check CFA info and insert CFI instructions if needed
; NO-O3-ENABLE-CFI-INSTR-INSERTER: Insert CFI remember/restore state instructions

; O0-WITHOUT-ENABLE-CFI-INSTR-INSERTER-LABEL: Pass Arguments:
; NO-O0-WITHOUT-ENABLE-CFI-INSTR-INSERTER: Check CFA info and insert CFI instructions if needed
; O0-WITHOUT-ENABLE-CFI-INSTR-INSERTER: Insert CFI remember/restore state instructions

; O0-ENABLE-CFI-INSTR-INSERTER-LABEL: Pass Arguments:
; O0-ENABLE-CFI-INSTR-INSERTER: Check CFA info and insert CFI instructions if needed
; NO-O0-ENABLE-CFI-INSTR-INSERTER: Insert CFI remember/restore state instructions
