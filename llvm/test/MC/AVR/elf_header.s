; RUN: llvm-mc -filetype=obj -triple avr -mcpu=at90s8515 %s -o - \
; RUN:     | llvm-readobj -h - | FileCheck --check-prefixes=ALL,AVR2 %s
; RUN: llvm-mc -filetype=obj -triple avr -mcpu=attiny13a %s -o - \
; RUN:     | llvm-readobj -h - | FileCheck --check-prefixes=ALL,AVR25 %s
; RUN: llvm-mc -filetype=obj -triple avr -mcpu=attiny167 %s -o - \
; RUN:     | llvm-readobj -h - | FileCheck --check-prefixes=ALL,AVR35 %s
; RUN: llvm-mc -filetype=obj -triple avr -mcpu=atmega88 %s -o - \
; RUN:     | llvm-readobj -h - | FileCheck --check-prefixes=ALL,AVR4 %s
; RUN: llvm-mc -filetype=obj -triple avr -mcpu=atmega16 %s -o - \
; RUN:     | llvm-readobj -h - | FileCheck --check-prefixes=ALL,AVR5 %s
; RUN: llvm-mc -filetype=obj -triple avr -mcpu=atmega128 %s -o - \
; RUN:     | llvm-readobj -h - | FileCheck --check-prefixes=ALL,AVR51 %s
; RUN: llvm-mc -filetype=obj -triple avr -mcpu=attiny817 %s -o - \
; RUN:     | llvm-readobj -h - | FileCheck --check-prefixes=ALL,XM3 %s
; RUN: llvm-mc -filetype=obj -triple avr -mcpu=atxmega256a3u %s -o - \
; RUN:     | llvm-readobj -h - | FileCheck --check-prefixes=ALL,XM6 %s
; RUN: llvm-mc -filetype=obj -triple avr -mcpu=atxmega256a3u %s -o - \
; RUN:     | llvm-readobj -h - | FileCheck --check-prefixes=ALL,XM6 %s
; RUN: llvm-mc -filetype=obj -triple avr -mcpu=attiny10 %s -o - \
; RUN:     | llvm-readobj -h - | FileCheck --check-prefixes=ALL,TINY %s

; ALL:       ElfHeader {
; ALL-NEXT:    Ident {
; ALL-NEXT:      Magic: (7F 45 4C 46)
; ALL-NEXT:      Class: 32-bit (0x1)
; ALL-NEXT:      DataEncoding: LittleEndian (0x1)
; ALL-NEXT:      FileVersion: 1
; ALL-NEXT:      OS/ABI: SystemV (0x0)
; ALL-NEXT:      ABIVersion: 0
; ALL-NEXT:      Unused: (00 00 00 00 00 00 00)
; ALL-NEXT:    }
; ALL-NEXT:    Type: Relocatable (0x1)
; ALL-NEXT:    Machine: EM_AVR (0x53)
; ALL-NEXT:    Version: 1
; ALL-NEXT:    Entry: 0x0
; ALL-NEXT:    ProgramHeaderOffset: 0x0
; ALL-NEXT:    SectionHeaderOffset: 0x5C

; AVR2:        Flags [ (0x82)
; AVR2-NEXT:     EF_AVR_ARCH_AVR2 (0x2)

; AVR25:       Flags [ (0x99)
; AVR25-NEXT:    EF_AVR_ARCH_AVR25 (0x19)

; AVR35:       Flags [ (0xA3)
; AVR35-NEXT:    EF_AVR_ARCH_AVR35 (0x23)

; AVR4:        Flags [ (0x84)
; AVR4-NEXT:     EF_AVR_ARCH_AVR4 (0x4)

; AVR5:        Flags [ (0x85)
; AVR5-NEXT:     EF_AVR_ARCH_AVR5 (0x5)

; AVR51:       Flags [ (0xB3)
; AVR51-NEXT:    EF_AVR_ARCH_AVR51 (0x33)

; XM3:         Flags [ (0xE7)
; XM3-NEXT:      EF_AVR_ARCH_XMEGA3 (0x67)

; XM6:         Flags [ (0xEA)
; XM6-NEXT:      EF_AVR_ARCH_XMEGA6 (0x6A)

; TINY:        Flags [ (0xE4)
; TINY-NEXT:     EF_AVR_ARCH_AVRTINY (0x64)

; ALL:           EF_AVR_LINKRELAX_PREPARED (0x80)
; ALL-NEXT:    ]
; ALL-NEXT:    HeaderSize: 52
; ALL-NEXT:    ProgramHeaderEntrySize: 0
; ALL-NEXT:    ProgramHeaderCount: 0
; ALL-NEXT:    SectionHeaderEntrySize: 40
; ALL-NEXT:    SectionHeaderCount: 4
; ALL-NEXT:    StringTableSectionIndex: 1
; ALL-NEXT:  }
