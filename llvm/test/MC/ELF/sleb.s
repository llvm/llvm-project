// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu %s -o - | llvm-readobj -S --sd | FileCheck -check-prefix=ELF_32 %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sd | FileCheck -check-prefix=ELF_64 %s
// RUN: llvm-mc -filetype=obj -triple i386-apple-darwin9 %s -o - | llvm-readobj -S --sd | FileCheck -check-prefix=MACHO_32 %s
// RUN: llvm-mc -filetype=obj -triple x86_64-apple-darwin9 %s -o - | llvm-readobj -S --sd | FileCheck -check-prefix=MACHO_64 %s

	.text
foo:
	.sleb128	0
	.sleb128	1
	.sleb128	-1
	.sleb128	63
	.sleb128	-64

	.sleb128	64
	.sleb128	-65

	.sleb128	8191
	.sleb128        -8192

	.sleb128        8193

// ELF_32:   Name: .text
// ELF_32:   SectionData (
// ELF_32:     0000: 00017F3F 40C000BF 7FFF3F80 4081C000
// ELF_32:   )
// ELF_64:   Name: .text
// ELF_64:   SectionData (
// ELF_64:     0000: 00017F3F 40C000BF 7FFF3F80 4081C000
// ELF_64:   )
// MACHO_32: Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// MACHO_32: SectionData (
// MACHO_32:   0000: 00017F3F 40C000BF 7FFF3F80 4081C000  |...?@.....?.@...|
// MACHO_32: )
// MACHO_64: Name: __text (5F 5F 74 65 78 74 00 00 00 00 00 00 00 00 00 00)
// MACHO_64: SectionData (
// MACHO_64:   0000: 00017F3F 40C000BF 7FFF3F80 4081C000  |...?@.....?.@...|
// MACHO_64: )
