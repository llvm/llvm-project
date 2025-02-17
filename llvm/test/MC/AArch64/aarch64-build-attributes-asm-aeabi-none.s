// RUN: llvm-mc -triple=aarch64 %s -o - | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF

// ASM: .aeabi_subsection   aeabi_pauthabi, required, uleb128
// ASM: .aeabi_attribute	1, 0 @ Tag_PAuth_Platform
// ASM: .aeabi_attribute	2, 0 @ Tag_PAuth_Schema
// ASM: .aeabi_subsection	aeabi_feature_and_bits, optional, uleb128
// ASM: .aeabi_attribute    0, 0 @ Tag_Feature_BTI
// ASM: .aeabi_attribute    1, 0 @ Tag_Feature_PAC
// ASM: .aeabi_attribute    2, 0 @ Tag_Feature_GCS

// ELF: Hex dump of section '.ARM.attributes':
// ELF-NEXT: 0x00000000 41190000 00616561 62695f70 61757468 A....aeabi_pauth
// ELF-NEXT: 0x00000010 61626900 00000100 02002300 00006165 abi.......#...ae
// ELF-NEXT: 0x00000020 6162695f 66656174 7572655f 616e645f abi_feature_and_
// ELF-NEXT: 0x00000030 62697473 00010000 00010002 00


.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 0
.aeabi_attribute Tag_PAuth_Schema, 0
.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 0
.aeabi_attribute Tag_Feature_PAC, 0
.aeabi_attribute Tag_Feature_GCS, 0
