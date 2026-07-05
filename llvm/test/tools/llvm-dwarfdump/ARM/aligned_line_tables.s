// RUN: llvm-mc %s -defsym ALIGN_4=1 -save-temp-labels -filetype obj -triple arm-none-eabi -o %t.o
// RUN: llvm-nm %t.o | FileCheck %s --check-prefix=L4
// RUN: llvm-dwarfdump -debug-line %t.o 2>&1 | FileCheck %s --implicit-check-not='warning:' --check-prefix=MULT4

// RUN: llvm-mc %s -defsym ALIGN_8=1 -save-temp-labels -filetype obj -triple arm-none-eabi -o %t.o
// RUN: llvm-nm %t.o | FileCheck %s --check-prefix=L8
// RUN: llvm-dwarfdump -debug-line %t.o 2>&1 | FileCheck %s --implicit-check-not='warning:' --check-prefix=MULT8

// RUN: llvm-mc %s -defsym UNALIGNED_PADDING=1 -save-temp-labels -filetype obj -triple arm-none-eabi -o %t.o
// RUN: llvm-nm %t.o | FileCheck %s --check-prefix=LUNALIGN
// RUN: llvm-dwarfdump -debug-line %t.o 2>&1 | FileCheck %s --check-prefix=UNALIGN

/// This test is based on a real example from ARM C/C++ Compiler.
/// It verifies llvm-dwarfdump is able to dump line tables even if they've been
/// placed at aligned offsets.

// L4: 0000002b N .Ltable0_end
// MULT4:      Address            Line   Column File   ISA Discriminator OpIndex Flags
// MULT4-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
// MULT4-NEXT: 0x0000000000000000      1      0      1   0             0       0  is_stmt end_sequence
// MULT4-EMPTY:
// MULT4-NEXT: debug_line[0x0000002c]
// MULT4-NEXT: Line table prologue:
// MULT4-NEXT:    total_length: 0x0000003a{{$}}
// MULT4-NEXT:          format: DWARF32
// MULT4-NEXT:         version: 2{{$}}
// MULT4-NEXT: prologue_length: 0x0000001a
// MULT4-NEXT: min_inst_length: 2
// MULT4-NEXT: default_is_stmt: 1

// L8: 00000027 N .Ltable0_end
// MULT8:      Address            Line   Column File   ISA Discriminator OpIndex Flags
// MULT8-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
// MULT8-NEXT: 0x0000000000000000      1      0      1   0             0       0  is_stmt end_sequence
// MULT8-EMPTY:
// MULT8-NEXT: debug_line[0x00000028]
// MULT8-NEXT: Line table prologue:
// MULT8-NEXT:    total_length: 0x0000003a{{$}}
// MULT8-NEXT:          format: DWARF32
// MULT8-NEXT:         version: 2{{$}}
// MULT8-NEXT: prologue_length: 0x0000001a
// MULT8-NEXT: min_inst_length: 2
// MULT8-NEXT: default_is_stmt: 1

/// This should fail to dump:
// LUNALIGN: 00000027 N .Ltable0_end
// UNALIGN: warning: parsing line table prologue at offset 0x00000027: unsupported version

.section .debug_line
/// First line table
/// Unit total length:
.long .Ltable0_end - .Ltable0_start
.Ltable0_start:
.short 2        /// Version
/// Header length:
.long .Ltable0_header_end - .Ltable0_header_start
.Ltable0_header_start:
.byte 4         /// Min instruction length
.byte 1         /// Max operations per instruction
.byte 0         /// Default is statement
.byte 6         /// Line range
.byte 10        /// Opcode base
.byte 0         /// standard_opcode_lengths[DW_LNS_copy] = 0
.byte 1         /// standard_opcode_lengths[DW_LNS_advance_pc] = 1
.byte 1         /// standard_opcode_lengths[DW_LNS_advance_line] = 1
.byte 1         /// standard_opcode_lengths[DW_LNS_set_file] = 1
.byte 1         /// standard_opcode_lengths[DW_LNS_set_column] = 1
.byte 0         /// standard_opcode_lengths[DW_LNS_negate_stmt] = 0
.byte 0         /// standard_opcode_lengths[DW_LNS_set_basic_block] = 0
.byte 0         /// standard_opcode_lengths[DW_LNS_const_add_pc] = 0
.byte 0         /// standard_opcode_lengths[DW_LNS_fixed_advance_pc] = 0
.byte 0         /// No include directories
/// File name:
.ifdef ALIGN_4
/// Pad out filename so next 4 byte aligned offset is a multiple of 4 and not 8.
.asciz "foobar.cpp"
.else
.asciz "test.c"
.endif
.byte 0         /// Dir idx
.byte 0         /// Mod time
.byte 0         /// Length
.byte 0         /// End files
.Ltable0_header_end:
/// Line table operations
.byte 0         /// Extended opcode
.byte 1         /// Length 1
.byte 1         /// DW_LNE_end_sequence
.Ltable0_end:
/// End first line table
/// Padding:
.ifdef UNALIGNED_PADDING
.short 0
.else
.byte 0
.endif
/// Second line table
/// Unit total length:
.long .Ltable1_end - .Ltable1_start
.Ltable1_start:
.short 2        /// Version
/// Header length:
.long .Ltable1_header_end - .Ltable1_header_start
.Ltable1_header_start:
.byte 2         /// Min instruction length
.byte 1         /// Max operations per instruction
.byte 0         /// Default is statement
.byte 6         /// Line range
.byte 10        /// Opcode base
.byte 0         /// standard_opcode_lengths[DW_LNS_copy] = 0
.byte 1         /// standard_opcode_lengths[DW_LNS_advance_pc] = 1
.byte 1         /// standard_opcode_lengths[DW_LNS_advance_line] = 1
.byte 1         /// standard_opcode_lengths[DW_LNS_set_file] = 1
.byte 1         /// standard_opcode_lengths[DW_LNS_set_column] = 1
.byte 0         /// standard_opcode_lengths[DW_LNS_negate_stmt] = 0
.byte 0         /// standard_opcode_lengths[DW_LNS_set_basic_block] = 0
.byte 0         /// standard_opcode_lengths[DW_LNS_const_add_pc] = 0
.byte 0         /// standard_opcode_lengths[DW_LNS_fixed_advance_pc] = 0
.byte 0         /// No include directories
.asciz "test.c" /// File name
.byte 0         /// Dir idx
.byte 0         /// Mod time
.byte 0         /// Length
.byte 0         /// End files
.Ltable1_header_end:
/// Line table operations
.byte 4         /// DW_LNS_set_file
.byte 1         /// File 1
.byte 5         /// DW_LNS_set_column
.byte 1         /// Column 1
.byte 0         /// Extended opcode
.byte 5         /// Length 5
.byte 2         /// DW_LNE_set_address
.long 32896     /// Address = 0x00008080
.byte 3         /// DW_LNS_advance_line
.byte 6         /// Line += 6
.byte 1         /// DW_LNS_copy
.byte 5         /// DW_LNS_set_column
.byte 2         /// Column 2
.byte 12        /// Special opcode (address += 0,  line += 2)
.byte 30        /// Special opcode (address += 6,  line += 2)
.byte 5         /// DW_LNS_set_column
.byte 1         /// Column 1
.byte 17        /// Special opcode (address += 2,  line += 1)
.byte 2         /// DW_LNS_advance_pc
.byte 4         /// += (4 * min instruction length)
.byte 0         /// Extended opcode
.byte 1         /// Length 1
.byte 1         /// DW_LNE_end_sequence
.Ltable1_end:
/// End second line table
.short 0        /// Padding (to make section a word multiple)
