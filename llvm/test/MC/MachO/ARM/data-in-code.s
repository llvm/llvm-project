@ RUN: llvm-mc -triple armv7-apple-darwin10 -filetype=obj -o - < %s | macho-dump | FileCheck %s
        .text
_foo:
@ CHECK: # DICE 0
@ CHECK: ('offset', 0)
@ CHECK: ('length', 4)
@ CHECK: ('kind', 1)
@ CHECK: # DICE 1
@ CHECK: ('offset', 4)
@ CHECK: ('length', 4)
@ CHECK: ('kind', 4)
@ CHECK: # DICE 2
@ CHECK: ('offset', 8)
@ CHECK: ('length', 2)
@ CHECK: ('kind', 3)
@ CHECK: # DICE 3
@ CHECK: ('offset', 10)
@ CHECK: ('length', 1)
@ CHECK: ('kind', 2)

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

