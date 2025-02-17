// RUN: llvm-mc -triple=aarch64 %s -o - | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF

// ASM: .aeabi_subsection aeabi_pauthabi, required, uleb128
// ASM: .aeabi_attribute 1, 7 @ Tag_PAuth_Platform
// ASM: .aeabi_attribute 2, 777 @ Tag_PAuth_Schema
// ASM: .aeabi_attribute 2, 777 @ Tag_PAuth_Schema
// ASM: .aeabi_subsection aeabi_feature_and_bits, optional, uleb128
// ASM: .aeabi_attribute 0, 1 @ Tag_Feature_BTI
// ASM: .aeabi_attribute 1, 1 @ Tag_Feature_PAC
// ASM: .aeabi_attribute 2, 1 @ Tag_Feature_GCS
// ASM: .aeabi_subsection aeabi_pauthabi, required, uleb128
// ASM: .aeabi_attribute 1, 7 @ Tag_PAuth_Platform
// ASM: .aeabi_attribute 2, 777 @ Tag_PAuth_Schema
// ASM: .aeabi_attribute 2, 777 @ Tag_PAuth_Schema
// ASM: .aeabi_subsection aeabi_feature_and_bits, optional, uleb128
// ASM: .aeabi_attribute 0, 1 @ Tag_Feature_BTI
// ASM: .aeabi_attribute 1, 1 @ Tag_Feature_PAC
// ASM: .aeabi_attribute 2, 1 @ Tag_Feature_GCS

// ELF: Hex dump of section '.ARM.attributes':
// ELF: 0x00000000 411a0000 00616561 62695f70 61757468 A....aeabi_pauth
// ELF: 0x00000010 61626900 00000107 02890623 00000061 abi........#...a
// ELF: 0x00000020 65616269 5f666561 74757265 5f616e64 eabi_feature_and
// ELF: 0x00000030 5f626974 73000100 00010101 0201     _bits.........


.aeabi_subsection aeabi_pauthabi, required, uleb128 @ test header comment
.aeabi_attribute Tag_PAuth_Platform, 7
.aeabi_attribute Tag_PAuth_Schema, 777
.aeabi_attribute Tag_PAuth_Schema, 777
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 1
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 1
.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute 1, 7 @ Tag_PAuth_Platform
.aeabi_attribute 2, 777 @ Tag_PAuth_Schema
.aeabi_attribute 2, 777 @ Tag_PAuth_Schema
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute 0, 1 @ Tag_Feature_BTI
.aeabi_attribute 1, 1 @ Tag_Feature_PAC
.aeabi_attribute 2, 1 @ Tag_Feature_GCS
