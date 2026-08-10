# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: llvm-dwarfdump -debug-gnu-pubnames %t.o | FileCheck %s

# CHECK: unit_offset = 0x00000000
# CHECK: unit_offset = 0x0000000c

	.section .debug_abbrev,"",@progbits
	.byte	1

	.section .debug_info,"",@progbits
.Lcu_begin0:
	.long	8           # Length of Unit
	.short	4           # DWARF version number
	.long	0           # Offset Into Abbrev. Section
	.byte	4           # Address Size
	.byte	0           # NULL
.Lcu_begin1:
	.long	8           # Length of Unit
	.short	4           # DWARF version number
	.long	0           # Offset Into Abbrev. Section
	.byte	4           # Address Size
	.byte	0           # NULL

	.section .debug_gnu_pubnames,"",@progbits
	.long	14
	.short	2           # DWARF Version
	.long	.Lcu_begin0
	.long	12          # Compilation Unit Length
	.long	0

	.long	14
	.short	2           # DWARF Version
	.long	.Lcu_begin1
	.long	12          # Compilation Unit Length
	.long	0
