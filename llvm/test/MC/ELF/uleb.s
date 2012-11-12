// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck -check-prefix=ELF_32 %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck -check-prefix=ELF_64 %s
// RUN: llvm-mc -filetype=obj -triple i386-apple-darwin9 %s -o - | macho-dump  --dump-section-data | FileCheck -check-prefix=MACHO_32 %s
// RUN: llvm-mc -filetype=obj -triple x86_64-apple-darwin9 %s -o - | macho-dump  --dump-section-data | FileCheck -check-prefix=MACHO_64 %s

	.text
foo:
	.uleb128	0
	.uleb128	1
	.uleb128	127
	.uleb128	128
	.uleb128	16383
	.uleb128	16384

// ELF_32: ('sh_name', 0x00000001) # '.text'
// ELF_32: ('_section_data', '00017f80 01ff7f80 8001')
// ELF_64: ('sh_name', 0x00000001) # '.text'
// ELF_64: ('_section_data', '00017f80 01ff7f80 8001')
// MACHO_32: ('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// MACHO_32: ('_section_data', '00017f80 01ff7f80 8001')
// MACHO_64: ('section_name', '__text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
// MACHO_64: ('_section_data', '00017f80 01ff7f80 8001')
