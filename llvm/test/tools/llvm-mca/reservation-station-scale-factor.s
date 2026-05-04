# REQUIRES: asserts
# REQUIRES: aarch64-registered-target

## Note: negative buffer size is not scaled 
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -reservation-station-scale-factor=2 -debug 2>&1 | FileCheck %s --check-prefix=UPSCALE
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -reservation-station-scale-factor=0.5 -debug 2>&1 | FileCheck %s --check-prefix=DOWNSCALE

## Default scale factor is 1
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -debug 2>&1 | FileCheck %s --check-prefix=ORIGINAL

## Negative scale factor is ignored
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -reservation-station-scale-factor=-1 -debug 2>&1 | FileCheck %s --check-prefix=ORIGINAL

## BufferSize=0 is not scaled
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=cortex-a53 -reservation-station-scale-factor=2 -debug 2>&1 | FileCheck %s --check-prefix=ZERO-BUFFER

# LLVM-MCA-BEGIN foo
add x2, x0, x1
# LLVM-MCA-END

# ORIGINAL-COUNT-1: Processor resources:
# ORIGINAL-NEXT: [ 0]  - 0x00000000000000 - InvalidUnit
# ORIGINAL-NEXT: [ 1]  - 0x00000000000001 - CyUnitB (BufferSize=24)
# ORIGINAL-NEXT: [ 2]  - 0x00000000000002 - CyUnitBR (BufferSize=-1)
# ORIGINAL-NEXT: [ 3]  - 0x00000000000004 - CyUnitFloatDiv (BufferSize=-1)
# ORIGINAL-NEXT: [ 4]  - 0x00000000000008 - CyUnitI (BufferSize=48)
# ORIGINAL-NEXT: [ 5]  - 0x00000000000010 - CyUnitID (BufferSize=16)
# ORIGINAL-NEXT: [ 6]  - 0x00000000000020 - CyUnitIM (BufferSize=32)
# ORIGINAL-NEXT: [ 7]  - 0x00000000000040 - CyUnitIS (BufferSize=24)
# ORIGINAL-NEXT: [ 8]  - 0x00000000000080 - CyUnitIntDiv (BufferSize=-1)
# ORIGINAL-NEXT: [ 9]  - 0x00000000000100 - CyUnitLS (BufferSize=28)
# ORIGINAL-NEXT: [10]  - 0x00000000000200 - CyUnitV (BufferSize=48)
# ORIGINAL-NEXT: [11]  - 0x00000000000400 - CyUnitVC (BufferSize=16)
# ORIGINAL-NEXT: [12]  - 0x00000000000800 - CyUnitVD (BufferSize=16)
# ORIGINAL-NEXT: [13]  - 0x00000000001000 - CyUnitVM (BufferSize=32)
# ORIGINAL:      [0] Code Region - foo

# DOWNSCALE-COUNT-1: Processor resources:
# DOWNSCALE-NEXT: [ 0]  - 0x00000000000000 - InvalidUnit
# DOWNSCALE-NEXT: [ 1]  - 0x00000000000001 - CyUnitB (BufferSize=12)
# DOWNSCALE-NEXT: [ 2]  - 0x00000000000002 - CyUnitBR (BufferSize=-1)
# DOWNSCALE-NEXT: [ 3]  - 0x00000000000004 - CyUnitFloatDiv (BufferSize=-1)
# DOWNSCALE-NEXT: [ 4]  - 0x00000000000008 - CyUnitI (BufferSize=24)
# DOWNSCALE-NEXT: [ 5]  - 0x00000000000010 - CyUnitID (BufferSize=8)
# DOWNSCALE-NEXT: [ 6]  - 0x00000000000020 - CyUnitIM (BufferSize=16)
# DOWNSCALE-NEXT: [ 7]  - 0x00000000000040 - CyUnitIS (BufferSize=12)
# DOWNSCALE-NEXT: [ 8]  - 0x00000000000080 - CyUnitIntDiv (BufferSize=-1)
# DOWNSCALE-NEXT: [ 9]  - 0x00000000000100 - CyUnitLS (BufferSize=14)
# DOWNSCALE-NEXT: [10]  - 0x00000000000200 - CyUnitV (BufferSize=24)
# DOWNSCALE-NEXT: [11]  - 0x00000000000400 - CyUnitVC (BufferSize=8)
# DOWNSCALE-NEXT: [12]  - 0x00000000000800 - CyUnitVD (BufferSize=8)
# DOWNSCALE-NEXT: [13]  - 0x00000000001000 - CyUnitVM (BufferSize=16)
# DOWNSCALE:      [0] Code Region - foo

# UPSCALE-COUNT-1: Processor resources:
# UPSCALE-NEXT: [ 0]  - 0x00000000000000 - InvalidUnit
# UPSCALE-NEXT: [ 1]  - 0x00000000000001 - CyUnitB (BufferSize=48)
# UPSCALE-NEXT: [ 2]  - 0x00000000000002 - CyUnitBR (BufferSize=-1)
# UPSCALE-NEXT: [ 3]  - 0x00000000000004 - CyUnitFloatDiv (BufferSize=-1)
# UPSCALE-NEXT: [ 4]  - 0x00000000000008 - CyUnitI (BufferSize=96)
# UPSCALE-NEXT: [ 5]  - 0x00000000000010 - CyUnitID (BufferSize=32)
# UPSCALE-NEXT: [ 6]  - 0x00000000000020 - CyUnitIM (BufferSize=64)
# UPSCALE-NEXT: [ 7]  - 0x00000000000040 - CyUnitIS (BufferSize=48)
# UPSCALE-NEXT: [ 8]  - 0x00000000000080 - CyUnitIntDiv (BufferSize=-1)
# UPSCALE-NEXT: [ 9]  - 0x00000000000100 - CyUnitLS (BufferSize=56)
# UPSCALE-NEXT: [10]  - 0x00000000000200 - CyUnitV (BufferSize=96)
# UPSCALE-NEXT: [11]  - 0x00000000000400 - CyUnitVC (BufferSize=32)
# UPSCALE-NEXT: [12]  - 0x00000000000800 - CyUnitVD (BufferSize=32)
# UPSCALE-NEXT: [13]  - 0x00000000001000 - CyUnitVM (BufferSize=64)
# UPSCALE:      [0] Code Region - foo

# ZERO-BUFFER-COUNT-1: Processor resources:
# ZERO-BUFFER-NEXT: [ 0]  - 0x00000000000000 - InvalidUnit
# ZERO-BUFFER-NEXT: [ 1]  - 0x00000000000001 - A53UnitALU (BufferSize=0)
# ZERO-BUFFER-NEXT: [ 2]  - 0x00000000000002 - A53UnitB (BufferSize=0)
# ZERO-BUFFER-NEXT: [ 3]  - 0x00000000000004 - A53UnitDiv (BufferSize=0)
# ZERO-BUFFER-NEXT: [ 4]  - 0x00000000000008 - A53UnitFPALU (BufferSize=0)
# ZERO-BUFFER-NEXT: [ 5]  - 0x00000000000010 - A53UnitFPMDS (BufferSize=0)
# ZERO-BUFFER-NEXT: [ 6]  - 0x00000000000020 - A53UnitLdSt (BufferSize=0)
# ZERO-BUFFER-NEXT: [ 7]  - 0x00000000000040 - A53UnitMAC (BufferSize=0)
# ZERO-BUFFER:      [0] Code Region - foo
