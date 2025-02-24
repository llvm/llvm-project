; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc %s -filetype=obj -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF

; ASM:      .aeabi_subsection	aeabi_feature_and_bits, optional, uleb128
; ASM-NEXT: .aeabi_attribute	0, 1 @ Tag_Feature_BTI
; ASM-NEXT: .aeabi_attribute	1, 0 @ Tag_Feature_PAC
; ASM-NEXT: .aeabi_attribute	2, 0 @ Tag_Feature_GCS

; ELF: Hex dump of section '.ARM.attributes':
; ELF-NEXT: 0x00000000 41230000 00616561 62695f66 65617475 A#...aeabi_featu
; ELF-NEXT: 0x00000010 72655f61 6e645f62 69747300 01000001 re_and_bits.....
; ELF-NEXT: 0x00000020 01000200 


target triple = "aarch64-unknown-none-elf"

!llvm.module.flags = !{!1}

!1 = !{i32 8, !"branch-target-enforcement", i32 1}
