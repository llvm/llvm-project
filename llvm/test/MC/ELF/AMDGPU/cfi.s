# RUN: llvm-mc -filetype=asm -mcpu=gfx900 -triple amdgcn-amd-amdhsa %s -o - | FileCheck --check-prefix=ASM %s
# RUN: llvm-mc -filetype=obj -mcpu=gfx900 -triple amdgcn-amd-amdhsa %s -o %t
# RUN: llvm-readelf -S -r -x .debug_frame %t | FileCheck --check-prefix=READELF %s

f:
	.cfi_sections .debug_frame
	.cfi_startproc
	s_nop 0
	.cfi_endproc

# ASM: f:
# ASM-NEXT: .cfi_sections .debug_frame
# ASM-NEXT: .cfi_startproc
# ASM-NEXT: s_nop 0
# ASM-NEXT: .cfi_endproc

# READELF: Section Headers:
# READELF: Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# READELF: .debug_frame      PROGBITS        0000000000000000 000048 000028 00      0   0  8

# READELF: Relocation section '.rela.debug_frame' at offset 0xd0 contains 2 entries:
# READELF-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# READELF-NEXT: 0000000000000014  0000000300000006 R_AMDGPU_ABS32         0000000000000000 .debug_frame + 0
# READELF-NEXT: 0000000000000018  0000000100000003 R_AMDGPU_ABS64         0000000000000000 .text + 0

# READELF: Hex dump of section '.debug_frame':
# READELF-NEXT: 0x00000000 0c000000 ffffffff 04000800 04041000 ................
# READELF-NEXT: 0x00000010 14000000 00000000 00000000 00000000 ................
# READELF-NEXT: 0x00000020 04000000 00000000                   ........
