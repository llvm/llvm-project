// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck -check-prefix=ELF_32 %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck -check-prefix=ELF_64 %s
// RUN: llvm-mc -filetype=obj -triple i386-apple-darwin9 %s -o - | macho-dump  --dump-section-data | FileCheck -check-prefix=MACHO_32 %s
// RUN: llvm-mc -filetype=obj -triple x86_64-apple-darwin9 %s -o - | macho-dump  --dump-section-data | FileCheck -check-prefix=MACHO_64 %s

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

// ELF_32: ('sh_name', 0x00000001) # '.text'
// ELF_32: ('_section_data', '00017f3f 40c000bf 7fff3f80 4081c000')
// ELF_64: ('sh_name', 0x00000001) # '.text'
// ELF_64: ('_section_data', '00017f3f 40c000bf 7fff3f80 4081c000')
// MACHO_32: ('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// MACHO_32: ('_section_data', '00017f3f 40c000bf 7fff3f80 4081c000')
// MACHO_64: ('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// MACHO_64: ('_section_data', '00017f3f 40c000bf 7fff3f80 4081c000')
