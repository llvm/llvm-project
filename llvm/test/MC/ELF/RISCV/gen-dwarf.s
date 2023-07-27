## Linker relaxation imposes restrictions on .eh_frame/.debug_frame, .debug_line,
## and LEB128 uses.

## CFI instructions can be preceded by relaxable instructions. We must use
## DW_CFA_advance_loc* opcodes with relocations.

## For .debug_line, emit DW_LNS_fixed_advance_pc with ADD16/SUB16 relocations so
## that .debug_line can be fixed by the linker. Without linker relaxation, we can
## emit special opcodes to make .debug_line smaller, but we don't do this for
## consistency.

# RUN: llvm-mc -filetype=obj -triple=riscv64 -g -dwarf-version=5 -mattr=+relax < %s -o %t
# RUN: llvm-dwarfdump -eh-frame -debug-line -debug-rnglists -v %t | FileCheck %s
# RUN: llvm-readobj -r -x .eh_frame %t | FileCheck %s --check-prefix=RELOC

# CHECK:      FDE
# CHECK-NEXT: Format:       DWARF32
# CHECK-NEXT: DW_CFA_advance_loc: 16
# CHECK-NEXT: DW_CFA_def_cfa_offset: +32
# CHECK-NEXT: DW_CFA_advance_loc: 4
# CHECK-NEXT: DW_CFA_offset: X1 -8
# CHECK-NEXT: DW_CFA_nop:

# CHECK:      DW_LNE_set_address
# CHECK-NEXT: DW_LNS_advance_line ([[#]])
# CHECK-NEXT: DW_LNS_copy
# CHECK-NEXT:                           is_stmt
# CHECK-NEXT: DW_LNS_advance_line
# CHECK-NEXT: DW_LNS_fixed_advance_pc (addr += 0x0004, op-index = 0)
# CHECK-NEXT: DW_LNS_copy
# CHECK-NEXT:                           is_stmt
# CHECK-NEXT: DW_LNS_advance_line
# CHECK-NEXT: DW_LNS_fixed_advance_pc (addr += 0x0004, op-index = 0)
# CHECK-NEXT: DW_LNS_copy

# CHECK:      0x00000000: range list header: length = 0x0000001d, format = DWARF32, version = 0x0005
# CHECK-NEXT: ranges:
# CHECK-NEXT: 0x0000000c: [DW_RLE_start_length]:  0x0000000000000000, 0x0000000000000034
# CHECK-NEXT: 0x00000016: [DW_RLE_start_length]:  0x0000000000000000, 0x0000000000000004
# CHECK-NEXT: 0x00000020: [DW_RLE_end_of_list ]

# RELOC:      Section ([[#]]) .rela.eh_frame {
# RELOC-NEXT:   0x1C R_RISCV_32_PCREL <null> 0x0
# RELOC-NEXT:   0x20 R_RISCV_ADD32 <null> 0x0
# RELOC-NEXT:   0x20 R_RISCV_SUB32 <null> 0x0
# RELOC-NEXT:   0x25 R_RISCV_SET6 <null> 0x0
# RELOC-NEXT:   0x25 R_RISCV_SUB6 <null> 0x0
# RELOC-NEXT:   0x34 R_RISCV_32_PCREL <null> 0x0
# RELOC-NEXT: }

## TODO A section needs two relocations.
# RELOC:      Section ([[#]]) .rela.debug_rnglists {
# RELOC-NEXT:   0xD R_RISCV_64 .text.foo 0x0
# RELOC-NEXT:   0x17 R_RISCV_64 .text.bar 0x0
# RELOC-NEXT: }

# RELOC:      Section ([[#]]) .rela.debug_line {
# RELOC:        R_RISCV_ADD16 <null> 0x0
# RELOC-NEXT:   R_RISCV_SUB16 <null> 0x0
# RELOC-NEXT:   R_RISCV_ADD16 <null> 0x0
# RELOC-NEXT:   R_RISCV_SUB16 <null> 0x0
# RELOC-NEXT:   R_RISCV_ADD16 <null> 0x0
# RELOC-NEXT:   R_RISCV_SUB16 <null> 0x0
# RELOC:      }

# RELOC:      Hex dump of section '.eh_frame':
# RELOC-NEXT: 0x00000000
# RELOC-NEXT: 0x00000010
# RELOC-NEXT: 0x00000020
# RELOC-NEXT: 0x00000030 30000000 00000000 04000000 00000000
#                                          ^ address_range

.section .text.foo,"ax"
.globl foo
foo:
.cfi_startproc
.Lpcrel_hi0:
  auipc a1, %pcrel_hi(g)
  lw a1, %pcrel_lo(.Lpcrel_hi0)(a1)
  bge a1, a0, .LBB0_2
  addi sp, sp, -32
  .cfi_def_cfa_offset 32
  sd ra, 24(sp)
  .cfi_offset ra, -8
  addi a0, sp, 8
  call ext@plt
  ld ra, 24(sp)
  addi sp, sp, 32
  ret
.LBB0_2:
  li a0, 0
  ret
  .cfi_endproc
  .size foo, .-foo

.section .text.bar,"ax"
bar:
.cfi_startproc
  nop
.cfi_endproc
