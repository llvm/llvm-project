# RUN: llvm-mc -arch=mips -mcpu=mips32r2 -filetype=obj %s -o - | \
# RUN:   llvm-readobj -sections -section-data - | \
# RUN:     FileCheck %s
	mfc0	$16, $15, 1
	mfc0	$16, $16, 1


# Checking for the coprocessor 0's register usage was recorded
# and emitted.
# CHECK:  Section {
# CHECK:     Index: 5
# CHECK:     Name: .reginfo (27)
# CHECK:     Type: SHT_MIPS_REGINFO (0x70000006)
# CHECK:     Flags [ (0x2)
# CHECK:       SHF_ALLOC (0x2)
# CHECK:     ]
# CHECK:     Address: 0x0
# CHECK:     Offset: 0x50
# CHECK:     Size: 24
# CHECK:     Link: 0
# CHECK:     Info: 0
# CHECK:     AddressAlignment: 4
# CHECK:     EntrySize: 24
# CHECK:     SectionData (
# CHECK:       0000: 00010000 00018000 00000000 00000000  |................|
# CHECK:       0010: 00000000 00000000                    |........|
# CHECK:     )
# CHECK:   }
