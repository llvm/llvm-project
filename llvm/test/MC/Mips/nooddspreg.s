# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -mattr=+fp64 | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -mattr=+fp64 -filetype=obj -o - | \
# RUN:   llvm-readobj --sections --section-data --section-relocations - | \
# RUN:     FileCheck %s -check-prefix=CHECK-OBJ

# RUN: not llvm-mc %s -triple mips64-unknown-linux-gnuabin32 2> %t1
# RUN: FileCheck %s -check-prefix=INVALID < %t1
#
# RUN: not llvm-mc %s -triple mips64-unknown-linux-gnu 2> %t2
# RUN: FileCheck %s -check-prefix=INVALID < %t2
#
# CHECK-ASM: .module nooddspreg

# Checking if the Mips.abiflags were correctly emitted.
# CHECK-OBJ:       Section {
# CHECK-OBJ:         Index: 5
# CHECK-OBJ-LABEL:   Name: .MIPS.abiflags (12)
# CHECK-OBJ:         Type: SHT_MIPS_ABIFLAGS (0x7000002A)
# CHECK-OBJ:          Flags [ (0x2)
# CHECK-OBJ:           SHF_ALLOC (0x2)
# CHECK-OBJ:         ]
# CHECK-OBJ:         Address: 0x0
# CHECK-OBJ:         Size: 24
# CHECK-OBJ:         Link: 0
# CHECK-OBJ:         Info: 0
# CHECK-OBJ:         AddressAlignment: 8
# CHECK-OBJ:         EntrySize: 24
# CHECK-OBJ:         Relocations [
# CHECK-OBJ:         ]
# CHECK-OBJ:         SectionData (
# CHECK-OBJ:           0000: 00002001 01020007 00000000 00000000  |.. .............|
# CHECK-OBJ:           0010: 00000000 00000000                    |........|
# CHECK-OBJ:         )
# CHECK-OBJ-LABEL: }

# INVALID: '.module nooddspreg' requires the O32 ABI

        .module nooddspreg

# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
