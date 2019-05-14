# RUN: llvm-mc -filetype=obj -triple mips64eb-unknown-freebsd %s -o - | llvm-readobj -S --sd | FileCheck %s

        .section        .fixups,"",@progbits
        .byte   0xff
$diff0 = ($loc1)-($loc0)
        .2byte   ($diff0)

        .byte   0xff
$diff1 = ($loc2)-($loc0)
        .4byte  ($diff1)

        .byte   0xff
$diff2 = ($loc3)-($loc0)
        .8byte  ($diff2)
        .byte   0xff

$loc0:
        .byte   0xee
$loc1:
        .byte   0xdd
$loc2:
        .byte   0xcc
$loc3:

# CHECK:	AddressSize: 64bit
# CHECK:	  Section {
# CHECK:	    Name: .fixups (12)
# CHECK-NEXT:	    Type: SHT_PROGBITS (0x1)
# CHECK-NEXT:	    Flags [ (0x0)
# CHECK-NEXT:	    ]
# CHECK-NEXT:	    Address: 0x0
# CHECK-NEXT:	    Offset: 0x40
# CHECK-NEXT:	    Size: 21
# CHECK-NEXT:	    Link: 0
# CHECK-NEXT:	    Info: 0
# CHECK-NEXT:	    AddressAlignment: 1
# CHECK-NEXT:	    EntrySize: 0
# CHECK-NEXT:	    SectionData (
# CHECK-NEXT:	      0000: FF0001FF 00000002 FF000000 00000000  |................|
# CHECK-NEXT:	      0010: 03FFEEDD CC                          |.....|
# CHECK-NEXT:	    )
# CHECK-NEXT:	  }
# CHECK:	]
