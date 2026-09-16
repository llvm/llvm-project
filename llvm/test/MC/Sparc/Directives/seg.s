! RUN: llvm-mc -triple=sparc -filetype=obj %s -o - \
! RUN:   | llvm-readobj -S --sd - | FileCheck %s
! RUN: llvm-mc -triple=sparcv9 -filetype=obj %s -o - \
! RUN:   | llvm-readobj -S --sd - | FileCheck %s
! RUN: not llvm-mc -triple=sparc --defsym ERR=1 %s -o /dev/null 2>&1 \
! RUN:   | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

! CHECK:      Name: .text
! CHECK:      SectionData (
! CHECK-NEXT:   0000: 01

! CHECK:      Name: .data
! CHECK:      SectionData (
! CHECK-NEXT:   0000: 02030405

! CHECK:      Name: .bss
! CHECK-NEXT: Type: SHT_NOBITS
! CHECK:      Size: 6

.seg "data1"
.byte 4
.seg "data"
.byte 2
.seg "text"
.byte 1
.seg "data1"
.byte 5
.seg "data"
.byte 3
.seg "bss"
.zero 6

.ifdef ERR
! ERR: :[[#@LINE+1]]:1: error: unknown segment type
.seg "rodata"
.endif
