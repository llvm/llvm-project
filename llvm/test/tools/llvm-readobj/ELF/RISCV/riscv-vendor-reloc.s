# RUN: llvm-mc -triple riscv32 %s -filetype=obj -o %t32.o
# RUN: llvm-mc -triple riscv64 %s -filetype=obj -o %t64.o
# RUN: llvm-readelf -r %t32.o | FileCheck %s --check-prefix=GNU
# RUN: llvm-readelf -r %t64.o | FileCheck %s --check-prefix=GNU
# RUN: llvm-readobj -r %t32.o | FileCheck %s --check-prefix=LLVM
# RUN: llvm-readobj -r %t64.o | FileCheck %s --check-prefix=LLVM

## Test that llvm-readelf/llvm-readobj correctly resolves RISCV vendor-specific
## relocation names when preceded by R_RISCV_VENDOR, and falls back to
## R_RISCV_CUSTOM* when there is no preceding vendor relocation or the vendor
## is unknown.

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

# GNU:      Relocation section '.rela.text'
# GNU:      R_RISCV_VENDOR       {{.*}} QUALCOMM + 0
# GNU-NEXT: R_RISCV_QC_ABS20_U   {{.*}} foo + 0
# GNU-NEXT: R_RISCV_CUSTOM192    {{.*}} bar + 0
# GNU-NEXT: R_RISCV_VENDOR       {{.*}} ANDES + 0
# GNU-NEXT: R_RISCV_NDS_BRANCH_10 {{.*}} baz + 0
# GNU-NEXT: R_RISCV_VENDOR       {{.*}} UNKNOWN_VENDOR + 0
# GNU-NEXT: R_RISCV_CUSTOM200    {{.*}} qux + 0
# GNU-NEXT: R_RISCV_VENDOR       {{.*}} QUALCOMM + 0
# GNU-NEXT: R_RISCV_QC_E_BRANCH  {{.*}} quux + 0
## Test 6: Different offsets - not a valid pair
# GNU-NEXT: R_RISCV_VENDOR       {{.*}} QUALCOMM + 0
# GNU-NEXT: R_RISCV_CUSTOM193    {{.*}} barney + 0
## Test 7: Intervening relocation - not a valid pair
# GNU-NEXT: R_RISCV_VENDOR       {{.*}} QUALCOMM + 0
# GNU-NEXT: R_RISCV_32           {{.*}} snork + 0
# GNU-NEXT: R_RISCV_CUSTOM193    {{.*}} zot + 0

# LLVM:      Relocations [
# LLVM:        R_RISCV_VENDOR QUALCOMM 0x0
# LLVM-NEXT:   R_RISCV_QC_ABS20_U foo 0x0
# LLVM-NEXT:   R_RISCV_CUSTOM192 bar 0x0
# LLVM-NEXT:   R_RISCV_VENDOR ANDES 0x0
# LLVM-NEXT:   R_RISCV_NDS_BRANCH_10 baz 0x0
# LLVM-NEXT:   R_RISCV_VENDOR UNKNOWN_VENDOR 0x0
# LLVM-NEXT:   R_RISCV_CUSTOM200 qux 0x0
# LLVM-NEXT:   R_RISCV_VENDOR QUALCOMM 0x0
# LLVM-NEXT:   R_RISCV_QC_E_BRANCH quux 0x0
## Test 6: Different offsets - not a valid pair
# LLVM-NEXT:   R_RISCV_VENDOR QUALCOMM 0x0
# LLVM-NEXT:   R_RISCV_CUSTOM193 barney 0x0
## Test 7: Intervening relocation - not a valid pair
# LLVM-NEXT:   R_RISCV_VENDOR QUALCOMM 0x0
# LLVM-NEXT:   R_RISCV_32 snork 0x0
# LLVM-NEXT:   R_RISCV_CUSTOM193 zot 0x0
