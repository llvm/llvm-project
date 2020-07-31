# RUN: llvm-mc -triple i386-pc-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-readobj -S --sd < %t - | FileCheck %s
.section test1
.byte 1
.section test2
.byte 2
.previous
.byte 1
.section test2
.byte 2
.previous
.byte 1
.section test1
.byte 1
.previous
.byte 1
.section test2
.byte 2
.pushsection test3
.byte 3
.pushsection test4
.byte 4
.pushsection test5
.byte 5
.popsection
.byte 4
.popsection
.byte 3
.popsection
.byte 2
.pushsection test3
.byte 3
.pushsection test4
.byte 4
.previous
.byte 3
.popsection
.byte 3
.previous
.byte 2
.section test1
.byte 1
.popsection
.byte 2
.previous
.byte 1
.previous

# CHECK:      Sections [
# CHECK:        Section {
# CHECK:          Name: test1
# CHECK-NEXT:     Type: SHT_PROGBITS
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x34
# CHECK-NEXT:     Size: 7
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:       0000: 01010101 010101
# CHECK-NEXT:     )
# CHECK-NEXT:   }
# CHECK:        Section {
# CHECK:          Name: test2
# CHECK-NEXT:     Type: SHT_PROGBITS
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x3B
# CHECK-NEXT:     Size: 6
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:       0000: 02020202 0202
# CHECK-NEXT:     )
# CHECK-NEXT:   }
# CHECK:        Section {
# CHECK:          Name: test3
# CHECK-NEXT:     Type: SHT_PROGBITS
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x41
# CHECK-NEXT:     Size: 5
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:       0000: 03030303 03
# CHECK-NEXT:     )
# CHECK-NEXT:   }
# CHECK:        Section {
# CHECK:          Name: test4
# CHECK-NEXT:     Type: SHT_PROGBITS
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x46
# CHECK-NEXT:     Size: 3
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:       0000: 040404
# CHECK-NEXT:     )
# CHECK-NEXT:   }
# CHECK:        Section {
# CHECK:          Name: test5
# CHECK-NEXT:     Type: SHT_PROGBITS
# CHECK-NEXT:     Flags [ (0x0)
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x49
# CHECK-NEXT:     Size: 1
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:     SectionData (
# CHECK-NEXT:       0000: 05
# CHECK-NEXT:     )
# CHECK-NEXT:   }
