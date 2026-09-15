# RUN: llvm-mc -triple riscv32 %s -filetype=obj -o %t32.o
# RUN: llvm-mc -triple riscv64 %s -filetype=obj -o %t64.o

## Test llvm-objdump -r (relocation dump mode)
# RUN: llvm-objdump -r %t32.o | FileCheck %s --check-prefix=RELOC
# RUN: llvm-objdump -r %t64.o | FileCheck %s --check-prefix=RELOC

## Test llvm-objdump -dr (disassembly with relocations)
# RUN: llvm-objdump -dr %t32.o | FileCheck %s --check-prefix=DISASM
# RUN: llvm-objdump -dr %t64.o | FileCheck %s --check-prefix=DISASM

## Test that llvm-objdump correctly resolves RISCV vendor-specific relocation
## names when preceded by R_RISCV_VENDOR, and falls back to R_RISCV_CUSTOM*
## when there is no preceding vendor relocation or the vendor is unknown.

.text
  nop

## Test 1: Known vendor (QUALCOMM) - should resolve to vendor-specific name
  .reloc ., R_RISCV_VENDOR, QUALCOMM
  .reloc ., R_RISCV_CUSTOM192, foo
  nop

## Test 2: Vendor symbol is consumed after one use per RISC-V psABI.
## The second R_RISCV_CUSTOM192 without a preceding R_RISCV_VENDOR should
## remain as R_RISCV_CUSTOM192.
  .reloc ., R_RISCV_CUSTOM192, bar
  nop

## Test 3: Known vendor (ANDES) - should resolve to vendor-specific name
  .reloc ., R_RISCV_VENDOR, ANDES
  .reloc ., R_RISCV_CUSTOM241, baz
  nop

## Test 4: Unknown vendor - should fall back to R_RISCV_CUSTOM*
  .reloc ., R_RISCV_VENDOR, UNKNOWN_VENDOR
  .reloc ., R_RISCV_CUSTOM200, qux
  nop

## Test 5: Another known vendor after unknown - should work correctly
  .reloc ., R_RISCV_VENDOR, QUALCOMM
  .reloc ., R_RISCV_CUSTOM193, quux
  nop

## Test 6: Unpaired R_RISCV_VENDOR followed by R_RISCV_CUSTOM* at a different
## offset - should NOT be treated as a valid pair.
  .reloc . - 1, R_RISCV_VENDOR, QUALCOMM
  .reloc ., R_RISCV_CUSTOM193, barney
  nop

## Test 7: A non-R_RISCV_CUSTOM* relocation in between a vendor relocation pair
## breaks the pairing - R_RISCV_VENDOR must be immediately before the
## vendor-specific relocation per psABI.
  .reloc ., R_RISCV_VENDOR, QUALCOMM
  .reloc ., R_RISCV_32, snork
  .reloc ., R_RISCV_CUSTOM193, zot
  nop

# RELOC:      RELOCATION RECORDS FOR [.text]:
# RELOC:      R_RISCV_VENDOR       QUALCOMM
# RELOC-NEXT: R_RISCV_QC_ABS20_U   foo
# RELOC-NEXT: R_RISCV_CUSTOM192    bar
# RELOC-NEXT: R_RISCV_VENDOR       ANDES
# RELOC-NEXT: R_RISCV_NDS_BRANCH_10 baz
# RELOC-NEXT: R_RISCV_VENDOR       UNKNOWN_VENDOR
# RELOC-NEXT: R_RISCV_CUSTOM200    qux
# RELOC-NEXT: R_RISCV_VENDOR       QUALCOMM
# RELOC-NEXT: R_RISCV_QC_E_BRANCH  quux
## Test 6: Different offsets - not a valid pair
# RELOC-NEXT: R_RISCV_VENDOR       QUALCOMM
# RELOC-NEXT: R_RISCV_CUSTOM193    barney
## Test 7: Intervening relocation - not a valid pair
# RELOC-NEXT: R_RISCV_VENDOR       QUALCOMM
# RELOC-NEXT: R_RISCV_32           snork
# RELOC-NEXT: R_RISCV_CUSTOM193    zot

# DISASM:      R_RISCV_VENDOR       QUALCOMM
# DISASM-NEXT: R_RISCV_QC_ABS20_U   foo
# DISASM:      R_RISCV_CUSTOM192    bar
# DISASM:      R_RISCV_VENDOR       ANDES
# DISASM-NEXT: R_RISCV_NDS_BRANCH_10 baz
# DISASM:      R_RISCV_VENDOR       UNKNOWN_VENDOR
# DISASM-NEXT: R_RISCV_CUSTOM200    qux
# DISASM:      R_RISCV_VENDOR       QUALCOMM
# DISASM-NEXT: R_RISCV_QC_E_BRANCH  quux
## Test 6: Different offsets - not a valid pair
# DISASM:      R_RISCV_VENDOR       QUALCOMM
# DISASM:      R_RISCV_CUSTOM193    barney
## Test 7: Intervening relocation - not a valid pair
# DISASM:      R_RISCV_VENDOR       QUALCOMM
# DISASM-NEXT: R_RISCV_32           snork
# DISASM:      R_RISCV_CUSTOM193    zot
