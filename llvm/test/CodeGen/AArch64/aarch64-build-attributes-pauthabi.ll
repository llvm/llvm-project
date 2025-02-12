; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc %s -filetype=obj -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF

; ASM: .aeabi_subsection	aeabi_pauthabi, required, uleb128
; ASM-NEXT: .aeabi_attribute	Tag_PAuth_Platform, 2
; ASM-NEXT: .aeabi_attribute	Tag_PAuth_Schema, 31

; ELF: Hex dump of section '.ARM.attributes':
; ELF-NEXT: 0x00000000 41190000 00616561 62695f70 61757468 A....aeabi_pauth
; ELF-NEXT: 0x00000010 61626900 00000102 021f


target triple = "aarch64-unknown-none-elf"

!llvm.module.flags = !{!1, !2}

!1 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 2}
!2 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 31}
