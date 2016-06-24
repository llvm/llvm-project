# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -mattr=+fp64,+nooddspreg | \
# RUN:   FileCheck %s -check-prefix=CHECK-ASM
#
# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -mattr=+fp64,+nooddspreg -filetype=obj -o - | \
# RUN:   llvm-readobj -sections -section-data -section-relocations - | \
# RUN:     FileCheck %s -check-prefix=CHECK-OBJ

# RUN: not llvm-mc %s -arch=mips -mcpu=mips64 -target-abi n32 -mattr=+nooddspreg 2> %t0
# RUN: FileCheck %s -check-prefix=INVALID < %t0
#
# RUN: not llvm-mc %s -arch=mips -mcpu=mips64 -target-abi n64 -mattr=+nooddspreg 2> %t0
# RUN: FileCheck %s -check-prefix=INVALID < %t0
#
# CHECK-ASM-NOT: .module nooddspreg

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

# INVALID: ERROR: -mno-odd-spreg requires the O32 ABI

# FIXME: Test should include gnu_attributes directive when implemented.
#        An explicit .gnu_attribute must be checked against the effective
#        command line options and any inconsistencies reported via a warning.
