@ RUN: llvm-mc -triple armv7-apple-darwin10 -filetype=obj -o - < %s | llvm-readobj --macho-data-in-code - | FileCheck %s
        .text
_foo:
.data_region
        .long 10
.end_data_region
.data_region jt32
        .long 1
.end_data_region
.data_region jt16
        .short 2
.end_data_region
.data_region jt8
        .byte 3
.end_data_region

@ CHECK: File: <stdin>
@ CHECK: Format: Mach-O arm
@ CHECK: Arch: arm
@ CHECK: AddressSize: 32bit
@ CHECK: DataInCode {
@ CHECK:   Data offset: 300
@ CHECK:   Data size: 32
@ CHECK:   Data entries [
@ CHECK:     Entry {
@ CHECK:       Index: 0
@ CHECK:       Offset: 0
@ CHECK:       Length: 4
@ CHECK:       Kind: 1
@ CHECK:     }
@ CHECK:     Entry {
@ CHECK:       Index: 1
@ CHECK:       Offset: 4
@ CHECK:       Length: 4
@ CHECK:       Kind: 4
@ CHECK:     }
@ CHECK:     Entry {
@ CHECK:       Index: 2
@ CHECK:       Offset: 8
@ CHECK:       Length: 2
@ CHECK:       Kind: 3
@ CHECK:     }
@ CHECK:     Entry {
@ CHECK:       Index: 3
@ CHECK:       Offset: 10
@ CHECK:       Length: 1
@ CHECK:       Kind: 2
@ CHECK:     }
@ CHECK:   ]
@ CHECK: }
