// RUN: llvm-mc -triple=aarch64 %s -o - | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF

// ASM: .aeabi_subsection	subsection_a, optional, uleb128
// ASM: .aeabi_subsection	aeabi_subsection, optional, ntbs
// ASM: .aeabi_subsection	subsection_b, required, uleb128
// ASM: .aeabi_subsection	aeabi_pauthabi, required, uleb128
// ASM: .aeabi_attribute	1, 7 @ Tag_PAuth_Platform
// ASM: .aeabi_attribute	2, 777 @ Tag_PAuth_Schema
// ASM: .aeabi_subsection	aeabi_feature_and_bits, optional, uleb128
// ASM: .aeabi_attribute    0, 1 @ Tag_Feature_BTI
// ASM: .aeabi_attribute    1, 1 @ Tag_Feature_PAC
// ASM: .aeabi_attribute    2, 1 @ Tag_Feature_GCS
// ASM: .aeabi_subsection	aeabi_subsection, optional, ntbs
// ASM: .aeabi_attribute	5, "Value"
// ASM: .aeabi_subsection	subsection_b, required, uleb128
// ASM: .aeabi_attribute	6, 536
// ASM: .aeabi_subsection	subsection_a, optional, uleb128
// ASM: .aeabi_attribute	7, 11

// ELF: Hex dump of section '.ARM.attributes':
// ELF: 0x00000000 41150000 00737562 73656374 696f6e5f A....subsection_
// ELF: 0x00000010 61000100 070b2000 00006165 6162695f a..... ...aeabi_
// ELF: 0x00000020 73756273 65637469 6f6e0001 01052256 subsection...."V
// ELF: 0x00000030 616c7565 22001600 00007375 62736563 alue".....subsec
// ELF: 0x00000040 74696f6e 5f620000 00069804 1a000000 tion_b..........
// ELF: 0x00000050 61656162 695f7061 75746861 62690000 aeabi_pauthabi..
// ELF: 0x00000060 00010702 89062300 00006165 6162695f ......#...aeabi_
// ELF: 0x00000070 66656174 7572655f 616e645f 62697473 feature_and_bits
// ELF: 0x00000080 00010000 01010102 01                .........


.aeabi_subsection subsection_a, optional, uleb128
.aeabi_subsection aeabi_subsection, optional, ntbs
.aeabi_subsection subsection_b, required, uleb128
.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 7
.aeabi_attribute Tag_PAuth_Schema, 777
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1
.aeabi_subsection aeabi_subsection, optional, ntbs
.aeabi_attribute 5, "Value"
.aeabi_subsection subsection_b, required, uleb128
.aeabi_attribute 6, 536
.aeabi_subsection subsection_a, optional, uleb128
.aeabi_attribute 7, 11
