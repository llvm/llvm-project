# RUN: llvm-mc -triple mips-unknown-linux < %s -show-encoding -target-abi=o32 \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple mips64-unknown-linux < %s -show-encoding -target-abi=n32 \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple mips64-unknown-linux < %s -show-encoding -target-abi=n64 \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple mips-unknown-linux < %s -show-encoding -target-abi=o32 \
# RUN:     -filetype=obj | llvm-readobj -sections -section-data -r | \
# RUN:     FileCheck -check-prefix=OBJ-O32 %s
# RUN: llvm-mc -triple mips64-unknown-linux < %s -show-encoding -target-abi=n32 \
# RUN:     -filetype=obj | llvm-readobj -sections -section-data -r | \
# RUN:     FileCheck -check-prefix=OBJ-N32 %s
# RUN: llvm-mc -triple mips64-unknown-linux < %s -show-encoding -target-abi=n64 \
# RUN:     -filetype=obj | llvm-readobj -sections -section-data -r | \
# RUN:     FileCheck -check-prefix=OBJ-N64 %s
	.text
foo:
	.reloc 4, R_MIPS_NONE, foo   # ASM: .reloc 4, R_MIPS_NONE, foo
	.reloc 0, R_MIPS_NONE, foo+4 # ASM: .reloc 0, R_MIPS_NONE, foo+4
	.reloc 8, R_MIPS_32, foo+8   # ASM: .reloc 8, R_MIPS_32, foo+8
	nop
	nop
	nop
	.reloc 12, R_MIPS_NONE       # ASM: .reloc 12, R_MIPS_NONE{{$}}
        nop

# OBJ-O32-LABEL: Name: .text
# OBJ-O32:       0000: 00000000 00000000 00000008
# OBJ-O32-LABEL: }
# OBJ-O32-LABEL: Relocations [
# OBJ-O32:       0x0 R_MIPS_NONE .text 0x0
# OBJ-O32:       0x4 R_MIPS_NONE .text 0x0
# OBJ-O32:       0x8 R_MIPS_32 .text 0x0
# OBJ-O32:       0xC R_MIPS_NONE -   0x0

# OBJ-N32-LABEL: Name: .text
# OBJ-N32:       0000: 00000000 00000000 00000000
# OBJ-N32-LABEL: }
# OBJ-N32-LABEL: Relocations [

# OBJ-N32:       0x4 R_MIPS_NONE .text 0x0
# OBJ-N32:       0x0 R_MIPS_NONE .text 0x4
# OBJ-N32:       0x8 R_MIPS_32   .text 0x8
# OBJ-N32:       0xC R_MIPS_NONE -     0x0

# OBJ-N64-LABEL: Name: .text
# OBJ-N64:       0000: 00000000 00000000 00000000
# OBJ-N64-LABEL: }
# OBJ-N64-LABEL: Relocations [
# OBJ-N64:       0x4 R_MIPS_NONE/R_MIPS_NONE/R_MIPS_NONE .text 0x0
# OBJ-N64:       0x0 R_MIPS_NONE/R_MIPS_NONE/R_MIPS_NONE .text 0x4
# OBJ-N64:       0x8 R_MIPS_32/R_MIPS_NONE/R_MIPS_NONE .text 0x8
# OBJ-N64:       0xC R_MIPS_NONE/R_MIPS_NONE/R_MIPS_NONE -   0x0
