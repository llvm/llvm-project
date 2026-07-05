# RUN: llvm-mc -filetype=asm -triple x86_64-pc-linux-gnu %s -o - | FileCheck --check-prefix=ASM %s
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
# RUN: llvm-readelf -S -r -x .eh_frame %t | FileCheck --check-prefix=READELF %s

f:
	.cfi_startproc
	nop
	.cfi_llvm_def_aspace_cfa %rcx, 0, 6
	nop
	.cfi_endproc

# ASM: f:
# ASM-NEXT: .cfi_startproc
# ASM-NEXT: nop
# ASM-NEXT: .cfi_llvm_def_aspace_cfa %rcx, 0, 6
# ASM-NEXT: nop
# ASM-NEXT: .cfi_endproc

# READELF: Section Headers:
# READELF: Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# READELF: .eh_frame         X86_64_UNWIND   0000000000000000 000048 000030 00   A  0   0  8

# READELF: Relocation section '.rela.eh_frame' at offset 0xc0 contains 1 entries:
# READELF-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# READELF-NEXT: 0000000000000020  0000000100000002 R_X86_64_PC32          0000000000000000 .text + 0

# READELF: Hex dump of section '.eh_frame':
# READELF-NEXT: 0x00000000 14000000 00000000 017a5200 01781001
# READELF-NEXT: 0x00000010 1b0c0708 90010000 14000000 1c000000
# READELF-NEXT: 0x00000020 00000000 02000000 00413002 00060000
