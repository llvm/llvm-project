## A .debug_frame section whose second FDE has an invalid CFI instruction
## program.
# RUN: llvm-mc -triple x86_64-unknown-linux-gnu %s -filetype=obj -o %t.o

## Test that dumping a single (valid) entry does not touch the invalid program
## at all: the dump succeeds and nothing is reported.
# RUN: llvm-dwarfdump --debug-frame=0x4c %t.o 2>%t.one.err | FileCheck %s --check-prefix=ONE
# RUN: count 0 < %t.one.err

# ONE:       .debug_frame contents:
# ONE-NEXT:  0000004c 00000018 00000000 FDE cie=00000000 pc=00003000...00003020
# ONE-NEXT:    Format:       DWARF32
# ONE-NEXT:    DW_CFA_advance_loc: 1 to 0x3001
# ONE-NEXT:    DW_CFA_def_cfa_offset: +24
# ONE-NEXT:    DW_CFA_nop:

## Dumping the whole section reports the invalid program of the entry at 0x30
## and keeps dumping the entries around it.
# RUN: not llvm-dwarfdump --debug-frame %t.o 2>%t.all.err | FileCheck %s --check-prefix=ALL
# RUN: FileCheck %s --check-prefix=ERR --input-file=%t.all.err

# ALL:       .debug_frame contents:
# ALL:       00000000 00000010 ffffffff CIE
# ALL:       00000014 00000018 00000000 FDE cie=00000000 pc=00001000...00001020
# ALL:         DW_CFA_def_cfa_offset: +16
## The entry with the invalid program is still listed, with the instructions
## that could be decoded before the invalid opcode.
# ALL:       00000030 00000018 00000000 FDE cie=00000000 pc=00002000...00002020
# ALL:       0000004c 00000018 00000000 FDE cie=00000000 pc=00003000...00003020
# ALL:         DW_CFA_def_cfa_offset: +24

# ERR:       error: invalid extended CFI opcode 0x1a

	.section	.debug_frame,"",@progbits
.Lcie:
	.long	.Lcie_end-.Lcie_start   # Length
.Lcie_start:
	.long	0xffffffff              # CIE id
	.byte	1                       # Version
	.byte	0                       # Augmentation string
	.byte	1                       # Code alignment factor
	.byte	0x78                    # Data alignment factor (-8)
	.byte	16                      # Return address register
	.byte	0x0c, 0x07, 0x08        # DW_CFA_def_cfa reg7 +8
	.byte	0x90, 0x01              # DW_CFA_offset reg16 -8
	.byte	0, 0                    # DW_CFA_nop
.Lcie_end:

## An FDE with a valid CFI program.
	.long	.Lfde0_end-.Lfde0_start # Length
.Lfde0_start:
	.long	.Lcie                   # CIE pointer
	.quad	0x1000                  # Initial location
	.quad	0x20                    # Address range
	.byte	0x41                    # DW_CFA_advance_loc 1
	.byte	0x0e, 0x10              # DW_CFA_def_cfa_offset +16
	.byte	0                       # DW_CFA_nop
.Lfde0_end:

## An FDE whose CFI program uses an opcode that does not exist.
	.long	.Lfde1_end-.Lfde1_start # Length
.Lfde1_start:
	.long	.Lcie                   # CIE pointer
	.quad	0x2000                  # Initial location
	.quad	0x20                    # Address range
	.byte	0x1a                    # Invalid extended opcode
	.byte	0, 0, 0                 # DW_CFA_nop
.Lfde1_end:

## Another FDE with a valid CFI program.
	.long	.Lfde2_end-.Lfde2_start # Length
.Lfde2_start:
	.long	.Lcie                   # CIE pointer
	.quad	0x3000                  # Initial location
	.quad	0x20                    # Address range
	.byte	0x41                    # DW_CFA_advance_loc 1
	.byte	0x0e, 0x18              # DW_CFA_def_cfa_offset +24
	.byte	0                       # DW_CFA_nop
.Lfde2_end:
