// RUN: llvm-mc -dwarf-version 2 -triple  i686-pc-linux-gnu %s -filetype=obj -o - | llvm-readobj -r -s -sd | FileCheck --check-prefixes=DWARF,DWARF2 %s
// RUN: llvm-mc -dwarf-version 3 -triple  i686-pc-linux-gnu %s -filetype=obj -o - | llvm-readobj -r -s -sd | FileCheck --check-prefixes=DWARF,DWARF3 %s
// RUN: llvm-mc -dwarf-version 4 -triple  i686-pc-linux-gnu %s -filetype=obj -o - | llvm-readobj -r -s -sd | FileCheck --check-prefixes=DWARF,DWARF4 %s

// DWARF:    Name: .debug_line (11)
// DWARF-NEXT:    Type: SHT_PROGBITS (0x1)
// DWARF-NEXT:    Flags [ (0x0)
// DWARF-NEXT:    ]
// DWARF-NEXT:    Address: 0x0
// DWARF-NEXT:    Offset: 0x35
// DWARF2-NEXT:    Size: 51
// DWARF3-NEXT:    Size: 51
// DWARF4-NEXT:    Size: 52
// DWARF-NEXT:    Link: 0
// DWARF-NEXT:    Info: 0
// DWARF-NEXT:    AddressAlignment: 1
// DWARF-NEXT:    EntrySize: 0
// DWARF2-NEXT:    SectionData (
// DWARF2-NEXT:      0000: 2F000000 02001A00 00000101 FB0E0D00  |/...............|
// DWARF2-NEXT:      0010: 01010101 00000001 00000100 666F6F00  |............foo.|
// DWARF2-NEXT:      0020: 00000000 00050200 00000003 3F010201  |............?...|
// DWARF2-NEXT:      0030: 000101                               |...|
// DWARF2-NEXT:    )
// DWARF3-NEXT:    SectionData (
// DWARF3-NEXT:      0000: 2F000000 03001A00 00000101 FB0E0D00  |/...............|
// DWARF3-NEXT:      0010: 01010101 00000001 00000100 666F6F00  |............foo.|
// DWARF3-NEXT:      0020: 00000000 00050200 00000003 3F010201  |............?...|
// DWARF3-NEXT:      0030: 000101                               |...|
// DWARF3-NEXT:    )
// DWARF4-NEXT:    SectionData (
// DWARF4-NEXT:      0000: 30000000 04001B00 00000101 01FB0E0D  |0...............|
// DWARF4-NEXT:      0010: 00010101 01000000 01000001 00666F6F  |.............foo|
// DWARF4-NEXT:      0020: 00000000 00000502 00000000 033F0102  |.............?..|
// DWARF4-NEXT:      0030: 01000101                             |....|
// DWARF4-NEXT:    )

        .file   1 "foo"
        .loc    1 64 0
        nop
