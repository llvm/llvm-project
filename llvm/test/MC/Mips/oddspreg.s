# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -mattr=+fp64 | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -mattr=+fp64 -filetype=obj -o - | \
# RUN:   llvm-readobj --sections --section-data --section-relocations - | \
# RUN:     FileCheck %s -check-prefixes=CHECK-OBJ-ALL,CHECK-OBJ-O32
#
# RUN: llvm-mc %s -triple mips64-unknown-linux-gnuabin32 | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -triple mips64-unknown-linux-gnuabin32 -filetype=obj -o - | \
# RUN:   llvm-readobj --sections --section-data --section-relocations - | \
# RUN:     FileCheck %s -check-prefixes=CHECK-OBJ-ALL,CHECK-OBJ-N32

# RUN: llvm-mc %s -triple mips64-unknown-linux-gnu | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# Repeat the -filetype=obj tests but this time use an empty assembly file. The
# output should be unchanged.
# RUN: llvm-mc /dev/null -triple mips64-unknown-linux-gnu -filetype=obj -o - | \
# RUN:   llvm-readobj --sections --section-data --section-relocations - | \
# RUN:     FileCheck %s -check-prefixes=CHECK-OBJ-ALL,CHECK-OBJ-N64

# RUN: llvm-mc /dev/null -triple mips-unknown-linux-gnu -mattr=+fp64 -filetype=obj -o - | \
# RUN:   llvm-readobj --sections --section-data --section-relocations - | \
# RUN:     FileCheck %s -check-prefixes=CHECK-OBJ-ALL,CHECK-OBJ-O32
#
# RUN: llvm-mc /dev/null -triple mips64-unknown-linux-gnuabin32 -filetype=obj -o - | \
# RUN:   llvm-readobj --sections --section-data --section-relocations - | \
# RUN:     FileCheck %s -check-prefixes=CHECK-OBJ-ALL,CHECK-OBJ-N32

# RUN: llvm-mc /dev/null -triple mips64-unknown-linux-gnu -filetype=obj -o - | \
# RUN:   llvm-readobj --sections --section-data --section-relocations - | \
# RUN:     FileCheck %s -check-prefixes=CHECK-OBJ-ALL,CHECK-OBJ-N64

# CHECK-ASM: .module oddspreg

# Checking if the Mips.abiflags were correctly emitted.
# CHECK-OBJ-ALL:       Section {
# CHECK-OBJ-ALL:         Index: 5
# CHECK-OBJ-ALL-LABEL:   Name: .MIPS.abiflags ({{[0-9]+}})
# CHECK-OBJ-ALL:         Type: SHT_MIPS_ABIFLAGS (0x7000002A)
# CHECK-OBJ-ALL:          Flags [ (0x2)
# CHECK-OBJ-ALL:           SHF_ALLOC (0x2)
# CHECK-OBJ-ALL:         ]
# CHECK-OBJ-ALL:         Address: 0x0
# CHECK-OBJ-ALL:         Size: 24
# CHECK-OBJ-ALL:         Link: 0
# CHECK-OBJ-ALL:         Info: 0
# CHECK-OBJ-ALL:         AddressAlignment: 8
# CHECK-OBJ-ALL:         EntrySize: 24
# CHECK-OBJ-ALL:         Relocations [
# CHECK-OBJ-ALL:         ]
# CHECK-OBJ-ALL:         SectionData (
# CHECK-OBJ-O32:           0000: 00002001 01020006 00000000 00000000  |.. .............|
# CHECK-OBJ-O32:           0010: 00000001 00000000                    |........|
# CHECK-OBJ-N32:           0000: 00004001 02020001 00000000 00000000  |..@.............|
# CHECK-OBJ-N32:           0010: 00000001 00000000                    |........|
# CHECK-OBJ-N64:           0000: 00004001 02020001 00000000 00000000  |..@.............|
# CHECK-OBJ-N64:           0010: 00000001 00000000                    |........|
# CHECK-OBJ-ALL:         )
# CHECK-OBJ-ALL-LABEL: }

        .module oddspreg
        add.s $f3, $f1, $f5

# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
