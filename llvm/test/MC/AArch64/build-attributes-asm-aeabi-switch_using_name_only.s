// RUN: llvm-mc -triple=aarch64 %s -o - | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF

// ASM: .aeabi_subsection aeabi_pauthabi, required, uleb128
// ASM: .aeabi_subsection aeabi_feature_and_bits, optional, uleb128
// ASM: .aeabi_attribute	0, 1 // Tag_Feature_BTI
// ASM: .aeabi_attribute	2, 1 // Tag_PAuth_Schema
// ASM: .aeabi_attribute	1, 1 // Tag_PAuth_Platform
// ASM: .aeabi_attribute	2, 1 // Tag_Feature_GCS
// ASM: .aeabi_attribute	1, 0 // Tag_Feature_PAC
// ASM: .aeabi_attribute	7, 1
// ASM: .aeabi_attribute	7, 0

// ELF: Hex dump of section '.ARM.attributes':
// ELF-NEXT: 0x00000000 411b0000 00616561 62695f70 61757468 A....aeabi_pauth
// ELF-NEXT: 0x00000010 61626900 00000201 01010700 25000000 abi.........%...
// ELF-NEXT: 0x00000020 61656162 695f6665 61747572 655f616e aeabi_feature_an
// ELF-NEXT: 0x00000030 645f6269 74730001 00000102 01010007 d_bits..........
// ELF-NEXT: 0x00000040 01


.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_subsection aeabi_feature_and_bits
.aeabi_subsection aeabi_pauthabi
.aeabi_attribute Tag_PAuth_Schema, 1
.aeabi_subsection aeabi_pauthabi
.aeabi_attribute Tag_PAuth_Platform, 1
.aeabi_subsection aeabi_pauthabi
.aeabi_subsection aeabi_feature_and_bits
.aeabi_attribute Tag_Feature_GCS, 1
.aeabi_subsection aeabi_pauthabi
.aeabi_subsection aeabi_feature_and_bits
.aeabi_attribute Tag_Feature_PAC, 0
.aeabi_subsection aeabi_feature_and_bits
.aeabi_attribute 7, 1
.aeabi_subsection aeabi_pauthabi
.aeabi_attribute 7, 0
