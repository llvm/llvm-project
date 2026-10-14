# REQUIRES: asserts
# REQUIRES: aarch64-registered-target

# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -debug 2>&1 | FileCheck %s

# LLVM-MCA-BEGIN foo
add x2, x0, x1
# LLVM-MCA-END

## Print detailed processor resources information on simulation
# CHECK-COUNT-1: Processor resources:
# CHECK-NEXT: [ 0]  - 0x00000000000000 - InvalidUnit
# CHECK-NEXT: [ 1]  - 0x00000000000001 - CyUnitB (BufferSize=24)
# CHECK-NEXT: [ 2]  - 0x00000000000002 - CyUnitBR (BufferSize=-1)
# CHECK-NEXT: [ 3]  - 0x00000000000004 - CyUnitFloatDiv (BufferSize=-1)
# CHECK-NEXT: [ 4]  - 0x00000000000008 - CyUnitI (BufferSize=48)
# CHECK-NEXT: [ 5]  - 0x00000000000010 - CyUnitID (BufferSize=16)
# CHECK-NEXT: [ 6]  - 0x00000000000020 - CyUnitIM (BufferSize=32)
# CHECK-NEXT: [ 7]  - 0x00000000000040 - CyUnitIS (BufferSize=24)
# CHECK-NEXT: [ 8]  - 0x00000000000080 - CyUnitIntDiv (BufferSize=-1)
# CHECK-NEXT: [ 9]  - 0x00000000000100 - CyUnitLS (BufferSize=28)
# CHECK-NEXT: [10]  - 0x00000000000200 - CyUnitV (BufferSize=48)
# CHECK-NEXT: [11]  - 0x00000000000400 - CyUnitVC (BufferSize=16)
# CHECK-NEXT: [12]  - 0x00000000000800 - CyUnitVD (BufferSize=16)
# CHECK-NEXT: [13]  - 0x00000000001000 - CyUnitVM (BufferSize=32)
# CHECK:      [0] Code Region - foo

## Do not print mask-only information on simulation
# CHECK-NOT: Processor resource masks:
