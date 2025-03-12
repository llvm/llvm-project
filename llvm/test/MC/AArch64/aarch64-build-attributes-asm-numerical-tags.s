// RUN: llvm-mc -triple=aarch64 %s -o - | FileCheck %s --check-prefix=ASM

// ASM: .aeabi_subsection	aeabi_pauthabi, required, uleb128
// ASM: .aeabi_attribute	0, 1
// ASM: .aeabi_attribute	Tag_PAuth_Platform, 1
// ASM: .aeabi_attribute	Tag_PAuth_Schema, 1
// ASM: .aeabi_attribute	3, 1
// ASM: .aeabi_attribute	4, 1
// ASM: .aeabi_attribute	5, 1
// ASM: .aeabi_subsection	aeabi_feature_and_bits, optional, uleb128
// ASM: .aeabi_attribute	Tag_Feature_BTI, 1
// ASM: .aeabi_attribute	Tag_Feature_PAC, 1
// ASM: .aeabi_attribute	Tag_Feature_GCS, 1
// ASM: .aeabi_attribute	3, 1
// ASM: .aeabi_attribute	4, 1
// ASM: .aeabi_attribute	5, 1

// ELF: Hex dump of section '.ARM.attributes':
// ELF-NEXT: 0x00000000 41210000 00616561 62695f70 61757468 A!...aeabi_pauth
// ELF-NEXT: 0x00000010 61626900 00000001 01010201 03010401 abi.............
// ELF-NEXT: 0x00000020 05012900 00006165 6162695f 66656174 ..)...aeabi_feat
// ELF-NEXT: 0x00000030 7572655f 616e645f 62697473 00010000 ure_and_bits....
// ELF-NEXT: 0x00000040 01010102 01030104 010501


.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute	0, 1
.aeabi_attribute	1, 1
.aeabi_attribute	2, 1
.aeabi_attribute	3, 1
.aeabi_attribute	4, 1
.aeabi_attribute	5, 1
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute	0, 1
.aeabi_attribute	1, 1
.aeabi_attribute	2, 1
.aeabi_attribute	3, 1
.aeabi_attribute	4, 1
.aeabi_attribute	5, 1
