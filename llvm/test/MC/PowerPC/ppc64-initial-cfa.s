# RUN: llvm-mc -triple=powerpc64-unknown-linux-gnu -filetype=obj %s | \
# RUN: llvm-readobj -S --sr --sd - | FileCheck %s -check-prefix=STATIC -check-prefix=STATIC-BE
# RUN: llvm-mc -triple=powerpc64-unknown-linux-gnu -filetype=obj %s | \
# RUN: llvm-readobj -S --sr --sd - | FileCheck %s -check-prefix=PIC -check-prefix=PIC-BE
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj %s | \
# RUN: llvm-readobj -S --sr --sd - | FileCheck %s -check-prefix=STATIC -check-prefix=STATIC-LE
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj %s | \
# RUN: llvm-readobj -S --sr --sd - | FileCheck %s -check-prefix=PIC -check-prefix=PIC-LE

_proc:
        .cfi_startproc
        nop
        .cfi_endproc

# STATIC:      Section {
# STATIC:        Name: .eh_frame
# STATIC-NEXT:   Type: SHT_PROGBITS
# STATIC-NEXT:   Flags [ (0x2)
# STATIC-NEXT:     SHF_ALLOC
# STATIC-NEXT:   ]
# STATIC-NEXT:   Address:
# STATIC-NEXT:   Offset:
# STATIC-NEXT:   Size: 40
# STATIC-NEXT:   Link: 0
# STATIC-NEXT:   Info: 0
# STATIC-NEXT:   AddressAlignment: 8
# STATIC-NEXT:   EntrySize: 
# STATIC-NEXT:   Relocations [
# STATIC-NEXT:   ]
# STATIC-NEXT:   SectionData (
# STATIC-BE-NEXT:  0000: 00000010 00000000 017A5200 04784101
# STATIC-LE-NEXT:  0000: 10000000 00000000 017A5200 04784101
# STATIC-BE-NEXT:  0010: 1B0C0100 00000010 00000018 00000000
# STATIC-LE-NEXT:  0010: 1B0C0100 10000000 18000000 00000000
# STATIC-BE-NEXT:  0020: 00000004 00000000
# STATIC-LE-NEXT:  0020: 04000000 00000000
# STATIC-NEXT:   )
# STATIC-NEXT: }

# STATIC:      Section {
# STATIC:        Name: .rela.eh_frame
# STATIC-NEXT:   Type: SHT_RELA
# STATIC-NEXT:   Flags [
# STATIC-NEXT:     SHF_INFO_LINK
# STATIC-NEXT:   ]
# STATIC-NEXT:   Address:
# STATIC-NEXT:   Offset:
# STATIC-NEXT:   Size: 24
# STATIC-NEXT:   Link:
# STATIC-NEXT:   Info:
# STATIC-NEXT:   AddressAlignment: 8
# STATIC-NEXT:   EntrySize: 24
# STATIC-NEXT:   Relocations [
# STATIC-NEXT:     0x1C R_PPC64_REL32 .text 0x0
# STATIC-NEXT:   ]

# PIC:      Section {
# PIC:        Name: .eh_frame
# PIC-NEXT:   Type: SHT_PROGBITS
# PIC-NEXT:   Flags [ (0x2)
# PIC-NEXT:     SHF_ALLOC
# PIC-NEXT:   ]
# PIC-NEXT:   Address:
# PIC-NEXT:   Offset:
# PIC-NEXT:   Size: 40
# PIC-NEXT:   Link: 0
# PIC-NEXT:   Info: 0
# PIC-NEXT:   AddressAlignment: 8
# PIC-NEXT:   EntrySize: 0
# PIC-NEXT:   Relocations [
# PIC-NEXT:   ]
# PIC-NEXT:   SectionData (
# PIC-BE-NEXT:  0000: 00000010 00000000 017A5200 04784101
# PIC-LE-NEXT:  0000: 10000000 00000000 017A5200 04784101
# PIC-BE-NEXT:  0010: 1B0C0100 00000010 00000018 00000000
# PIC-LE-NEXT:  0010: 1B0C0100 10000000 18000000 00000000
# PIC-BE-NEXT:  0020: 00000004 00000000
# PIC-LE-NEXT:  0020: 04000000 00000000
# PIC-NEXT:   )
# PIC-NEXT: }

# PIC:      Section {
# PIC:        Name: .rela.eh_frame
# PIC-NEXT:   Type: SHT_RELA
# PIC-NEXT:   Flags [
# PIC-NEXT:     SHF_INFO_LINK
# PIC-NEXT:   ]
# PIC-NEXT:   Address:
# PIC-NEXT:   Offset:
# PIC-NEXT:   Size: 24
# PIC-NEXT:   Link:
# PIC-NEXT:   Info:
# PIC-NEXT:   AddressAlignment: 8
# PIC-NEXT:   EntrySize: 24
# PIC-NEXT:   Relocations [
# PIC-NEXT:     0x1C R_PPC64_REL32 .text 0x0
# PIC-NEXT:   ]
