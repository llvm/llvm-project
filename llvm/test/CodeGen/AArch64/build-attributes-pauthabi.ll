; RUN: rm -rf %t && split-file %s %t && cd %t

;--- platform-2-version-31.ll

; RUN: llc < platform-2-version-31.ll | FileCheck %s --check-prefix=ASM-2-31
; RUN: llc platform-2-version-31.ll -filetype=obj -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF-2-31

; ASM-2-31: .aeabi_subsection	aeabi_pauthabi, required, uleb128
; ASM-2-31-NEXT: .aeabi_attribute	1, 2 // Tag_PAuth_Platform
; ASM-2-31-NEXT: .aeabi_attribute	2, 31 // Tag_PAuth_Schema

; ELF-2-31: Hex dump of section '.ARM.attributes':
; ELF-2-31-NEXT: 0x00000000 41190000 00616561 62695f70 61757468 A....aeabi_pauth
; ELF-2-31-NEXT: 0x00000010 61626900 00000102 021f

target triple = "aarch64-unknown-none-elf"

!llvm.module.flags = !{!1, !2}

!1 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 2}
!2 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 31}


;--- platform-2-version-0.ll

; RUN: llc < platform-2-version-0.ll | FileCheck %s --check-prefix=ASM-2-0
; RUN: llc platform-2-version-0.ll -filetype=obj -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF-2-0

; ASM-2-0: .aeabi_subsection	aeabi_pauthabi, required, uleb128
; ASM-2-0-NEXT: .aeabi_attribute	1, 2 // Tag_PAuth_Platform
; ASM-2-0-NEXT: .aeabi_attribute	2, 0 // Tag_PAuth_Schema

; ELF-2-0: Hex dump of section '.ARM.attributes':
; ELF-2-0-NEXT: 0x00000000 41190000 00616561 62695f70 61757468 A....aeabi_pauth
; ELF-2-0-NEXT: 0x00000010 61626900 00000102 0200

target triple = "aarch64-unknown-none-elf"

!llvm.module.flags = !{!1, !2}

!1 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 2}
!2 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 0}


;--- platform-llvm_linux-version-31.ll

; RUN: llc < platform-llvm_linux-version-31.ll | FileCheck %s --check-prefix=ASM-LLVM_LINUX-31
; RUN: llc platform-llvm_linux-version-31.ll -filetype=obj -o - | llvm-readelf --hex-dump=.ARM.attributes - | FileCheck %s --check-prefix=ELF-LLVM_LINUX-31

; ASM-LLVM_LINUX-31: .aeabi_subsection	aeabi_pauthabi, required, uleb128
; ASM-LLVM_LINUX-31-NEXT: .aeabi_attribute	1, 268435458 // Tag_PAuth_Platform
; ASM-LLVM_LINUX-31-NEXT: .aeabi_attribute	2, 31 // Tag_PAuth_Schema

; ELF-LLVM_LINUX-31: Hex dump of section '.ARM.attributes':
; ELF-LLVM_LINUX-31-NEXT: 0x00000000 411d0000 00616561 62695f70 61757468 A....aeabi_pauth
; ELF-LLVM_LINUX-31-NEXT: 0x00000010 61626900 00000182 80808001 021f
target triple = "aarch64-unknown-none-elf"

!llvm.module.flags = !{!1, !2}

!1 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
!2 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 31}


;--- platform-llvm_linux-version-0.ll

; For LLVM_LINUX experimental platform, version value of 0 means no PAuth support.
; Do not emit corresponding PAuthABI GNU property note and AArch64 build attributes
; for this case to keep Linux binaries not using PAuth unaffected.

; RUN: llc < platform-llvm_linux-version-0.ll | FileCheck %s --check-prefix=ASM-LLVM_LINUX-0
; RUN: llc platform-llvm_linux-version-0.ll -filetype=obj -o - | llvm-readelf --sections - | FileCheck %s --check-prefix=ELF-LLVM_LINUX-0

; ASM-LLVM_LINUX-0-NOT: aeabi_pauthabi
; ELF-LLVM_LINUX-0-NOT: .ARM.attributes

target triple = "aarch64-unknown-none-elf"

!llvm.module.flags = !{!1, !2}

!1 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
!2 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 0}
