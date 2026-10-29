# REQUIRES: asserts
# REQUIRES: aarch64-registered-target

# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -debug -instruction-tables 2>&1 | FileCheck %s

# LLVM-MCA-BEGIN foo
add x2, x0, x1
# LLVM-MCA-END

## Do not print detailed processor resources information without simulation
# CHECK-NOT: Processor resources:

## Print mask-only information without simulation
# CHECK-COUNT-1: Processor resource masks:
# CHECK-NEXT: [ 0]  - 0x00000000000000 - InvalidUnit
# CHECK-NEXT: [ 1]  - 0x00000000000001 - CyUnitB
# CHECK-NEXT: [ 2]  - 0x00000000000002 - CyUnitBR
# CHECK-NEXT: [ 3]  - 0x00000000000004 - CyUnitFloatDiv
# CHECK-NEXT: [ 4]  - 0x00000000000008 - CyUnitI
# CHECK-NEXT: [ 5]  - 0x00000000000010 - CyUnitID
# CHECK-NEXT: [ 6]  - 0x00000000000020 - CyUnitIM
# CHECK-NEXT: [ 7]  - 0x00000000000040 - CyUnitIS
# CHECK-NEXT: [ 8]  - 0x00000000000080 - CyUnitIntDiv
# CHECK-NEXT: [ 9]  - 0x00000000000100 - CyUnitLS
# CHECK-NEXT: [10]  - 0x00000000000200 - CyUnitV
# CHECK-NEXT: [11]  - 0x00000000000400 - CyUnitVC
# CHECK-NEXT: [12]  - 0x00000000000800 - CyUnitVD
# CHECK-NEXT: [13]  - 0x00000000001000 - CyUnitVM
# CHECK:      [0] Code Region - foo
