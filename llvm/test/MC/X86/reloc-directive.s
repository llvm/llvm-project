# RUN: llvm-mc -triple i686-pc-win32 < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple i686-pc-win32 < %s -show-encoding \
# RUN:     -filetype=obj | llvm-readobj -sections -section-data -r | \
# RUN:     FileCheck -check-prefix=OBJ-32 %s
# RUN: llvm-mc -triple x86_64-pc-win32 < %s -show-encoding \
# RUN:     -filetype=obj | llvm-readobj -sections -section-data -r | \
# RUN:     FileCheck -check-prefix=OBJ-64 %s
	.text
foo:
	.long 0
	.long 0
	.long 0
	.long 0
	.long 0
	.reloc 4, dir32,    foo          # ASM: .reloc 4, dir32, foo
	.reloc 0, secrel32, foo+4        # ASM: .reloc 0, secrel32, foo+4
	.reloc 8, secidx,   foo+8        # ASM: .reloc 8, secidx, foo+8
	.reloc 12, dir32,   foo@secrel32 # ASM: .reloc 12, dir32, foo@SECREL32
	.reloc 16, dir32,   foo@imgrel   # ASM: .reloc 16, dir32, foo@IMGREL

# OBJ-32-LABEL: Name: .text
# OBJ-32:       0000: 04000000 00000000 00000000
# OBJ-32-LABEL: }
# OBJ-32-LABEL: Relocations [
# OBJ-32:       0x4  IMAGE_REL_I386_DIR32   foo
# OBJ-32:       0x0  IMAGE_REL_I386_SECREL  foo
# OBJ-32:       0x8  IMAGE_REL_I386_SECTION foo
# OBJ-32:       0xC  IMAGE_REL_I386_SECREL  foo
# OBJ-32:       0x10 IMAGE_REL_I386_DIR32NB foo

# OBJ-64-LABEL: Name: .text
# OBJ-64:       0000: 04000000 00000000 00000000
# OBJ-64-LABEL: }
# OBJ-64-LABEL: Relocations [
# OBJ-64:       0x4  IMAGE_REL_AMD64_ADDR32   foo
# OBJ-64:       0x0  IMAGE_REL_AMD64_SECREL   foo
# OBJ-64:       0x8  IMAGE_REL_AMD64_SECTION  foo
# OBJ-64:       0xC  IMAGE_REL_AMD64_SECREL   foo
# OBJ-64:       0x10 IMAGE_REL_AMD64_ADDR32NB foo
