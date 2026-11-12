// RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu    %s -o - | llvm-readobj -S --sd - | FileCheck %s -check-prefix=CHECK -check-prefix=ELF
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu  %s -o - | llvm-readobj -S --sd - | FileCheck %s -check-prefix=CHECK -check-prefix=ELF
// RUN: llvm-mc -filetype=obj -triple i386-apple-darwin9 --defsym MACHO=1 %s -o - | llvm-readobj -S --sd - | FileCheck %s -check-prefix=CHECK -check-prefix=MACHO
// RUN: llvm-mc -filetype=obj -triple x86_64-apple-darwin9 --defsym MACHO=1 %s -o - | llvm-readobj -S --sd - | FileCheck %s -check-prefix=CHECK -check-prefix=MACHO

// Test that we can assemble a GCC-like EH table that has 16381-16383 bytes of
// non-padding data between .ttbaseref and .ttbase. The assembler must insert
// extra padding either into the uleb128 or at the balign directive. See
// PR35809.

        .data
        .balign 4
foo:
        .byte 0xff  // LPStart omitted
        .byte 0x1   // TType encoding (uleb128)
.ifdef MACHO
        .uleb128 Lttbase-Lttbaseref
Lttbaseref:
.else
        .uleb128 .ttbase-.ttbaseref
.ttbaseref:
.endif
        .fill 128*128-1, 1, 0xcd    // call site and actions tables
        .balign 4
.ifdef MACHO
Lttbase:
.else
.ttbase:
.endif
        .byte 1, 2, 3, 4

// ELF:   Name: .data
// MACHO: Name: __data
// CHECK:      SectionData (
// CHECK-NEXT:   0000: FF01FFFF 00CDCDCD CDCDCDCD CDCDCDCD
// CHECK:        4000: CDCDCDCD 01020304
// CHECK-NEXT: )
