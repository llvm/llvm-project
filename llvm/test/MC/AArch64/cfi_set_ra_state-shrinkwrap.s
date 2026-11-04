# RUN: llvm-mc -triple arm64e-apple-darwin -mattr=+pauth-lr %s -filetype=obj -o %t.o
# RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t.o | FileCheck %s

# Vanilla ptrauth-returns=pauth frame.
  .globl _pauth
_pauth:
  .cfi_startproc
  .cfi_set_ra_state 1, 0
  pacibsp

  .cfi_set_ra_state 0, -4
  retab
  .cfi_endproc


# Vanilla ptrauth-returns=pauth-lr frame, with .cfi_set_ra_state instead of
# the deprecated .cfi_negate_ra_state_with_pc
  .globl _pauthlr
_pauthlr:
  .cfi_startproc
  .cfi_set_ra_state 2, 0
  pacibsppc

  .cfi_set_ra_state 0, -4
  retabsppc _pauthlr
  .cfi_endproc


# Shrink-wrapped frame, with a strange block ordering that puts the aut before
# the pac, which is not representable via .cfi_negate_ra_state_with_pc
# See: https://github.com/ARM-software/abi-aa/issues/327
  .globl _pauthlr_shrinkwrap
_pauthlr_shrinkwrap:
  .cfi_startproc
  cmp w0, #42
  b.lt LBB0
  ret

LBB1:
  ldp x29, x30, [sp], #16
  .cfi_def_cfa_offset 0
  .cfi_restore x29
  .cfi_restore x30

  adrp x16, LBB0@PAGE
  add x16, x16, LBB0@PAGEOFF
  .cfi_set_ra_state 0, 16
  autibsppcr x16

  mov w0, #42
  ret

LBB0:
  pacibsppc
  .cfi_set_ra_state 2, LBB0

  stp x29, x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset x29, -16
  .cfi_offset x30, -8

  b LBB1
  .cfi_endproc


# CHECK-LABEL:  Contents of __compact_unwind section:
# CHECK-NEXT:     Entry at offset 0x0:
# CHECK-NEXT:       start:                0x0 ltmp0
# CHECK-NEXT:       length:               0x8
# CHECK-NEXT:       compact encoding:     0x03000000
# CHECK-NEXT:     Entry at offset 0x20:
# CHECK-NEXT:       start:                0x8 _pauthlr
# CHECK-NEXT:       length:               0x8
# CHECK-NEXT:       compact encoding:     0x03000000
# CHECK-NEXT:     Entry at offset 0x40:
# CHECK-NEXT:       start:                0x10 _pauthlr_shrinkwrap
# CHECK-NEXT:       length:               0x30
# CHECK-NEXT:       compact encoding:     0x03000000
# CHECK-EMPTY:
# CHECK-LABEL:  .debug_frame contents:
# CHECK-EMPTY:
# CHECK-EMPTY:
# CHECK-LABEL:  .eh_frame contents:
# CHECK-EMPTY:
# CHECK-NEXT:   00000000 00000010 00000000 CIE
# CHECK-NEXT:     Format:                DWARF32
# CHECK-NEXT:     Version:               1
# CHECK-NEXT:     Augmentation:          "zR"
# CHECK-NEXT:     Code alignment factor: 1
# CHECK-NEXT:     Data alignment factor: -8
# CHECK-NEXT:     Return address column: 30
# CHECK-NEXT:     Augmentation data:     10
# CHECK-EMPTY:
# CHECK-NEXT:     DW_CFA_def_cfa: reg31 +0
# CHECK-EMPTY:
# CHECK-NEXT:     CFA=reg31
# CHECK-EMPTY:
# CHECK-NEXT:   00000014 00000020 00000018 FDE cie=00000000 pc=000000a0...000000a8
# CHECK-NEXT:     Format:       DWARF32
# CHECK-NEXT:     DW_CFA_AARCH64_set_ra_state: 1 0
# CHECK-NEXT:     DW_CFA_advance_loc: 4 to 0xa4
# CHECK-NEXT:     DW_CFA_AARCH64_set_ra_state: 0 -4
# CHECK-NEXT:     DW_CFA_nop:
# CHECK-NEXT:     DW_CFA_nop:
# CHECK-NEXT:     DW_CFA_nop:
# CHECK-NEXT:     DW_CFA_nop:
# CHECK-EMPTY:
# CHECK-NEXT:     0xa0: CFA=reg31: reg34=1
# CHECK-NEXT:     0xa4: CFA=reg31: reg34=0
# CHECK-EMPTY:
# CHECK-NEXT:   00000038 0000001c 0000003c FDE cie=00000000 pc=000000a0...000000a8
# CHECK-NEXT:     Format:       DWARF32
# CHECK-NEXT:     DW_CFA_AARCH64_set_ra_state: 2 0
# CHECK-NEXT:     DW_CFA_advance_loc: 4 to 0xa4
# CHECK-NEXT:     DW_CFA_AARCH64_set_ra_state: 0 -4
# CHECK-EMPTY:
# CHECK-NEXT:     0xa0: CFA=reg31: reg34=2
# CHECK-NEXT:     0xa4: CFA=reg31: reg34=0
# CHECK-EMPTY:
# CHECK-NEXT:   00000058 0000002c 0000005c FDE cie=00000000 pc=000000a0...000000d0
# CHECK-NEXT:     Format:       DWARF32
# CHECK-NEXT:     DW_CFA_advance_loc: 16 to 0xb0
# CHECK-NEXT:     DW_CFA_def_cfa_offset: +0
# CHECK-NEXT:     DW_CFA_restore: reg29
# CHECK-NEXT:     DW_CFA_restore: reg30
# CHECK-NEXT:     DW_CFA_advance_loc: 8 to 0xb8
# CHECK-NEXT:     DW_CFA_AARCH64_set_ra_state: 0 16
# CHECK-NEXT:     DW_CFA_advance_loc: 16 to 0xc8
# CHECK-NEXT:     DW_CFA_AARCH64_set_ra_state: 2 -4
# CHECK-NEXT:     DW_CFA_advance_loc: 4 to 0xcc
# CHECK-NEXT:     DW_CFA_def_cfa_offset: +16
# CHECK-NEXT:     DW_CFA_offset: reg29 -16
# CHECK-NEXT:     DW_CFA_offset: reg30 -8
# CHECK-NEXT:     DW_CFA_nop:
# CHECK-NEXT:     DW_CFA_nop:
# CHECK-NEXT:     DW_CFA_nop:
# CHECK-EMPTY:
# CHECK-NEXT:     0xa0: CFA=reg31
# CHECK-NEXT:     0xb0: CFA=reg31
# CHECK-NEXT:     0xb8: CFA=reg31: reg34=0
# CHECK-NEXT:     0xc8: CFA=reg31: reg34=2
# CHECK-NEXT:     0xcc: CFA=reg31+16: reg29=[CFA-16], reg30=[CFA-8], reg34=2
