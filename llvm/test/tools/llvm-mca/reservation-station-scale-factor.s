# REQUIRES: asserts
# REQUIRES: aarch64-registered-target

## Check upscaling and downscaling. BufferSize=-1 is not scaled
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -sched-model-reservation-station-scale-factor=2 -debug 2>&1 | FileCheck %s --check-prefix=UPSCALE
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -sched-model-reservation-station-scale-factor=1.5 -debug 2>&1 | FileCheck %s --check-prefix=UPSCALE-FRAC
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -sched-model-reservation-station-scale-factor=0.5 -debug 2>&1 | FileCheck %s --check-prefix=DOWNSCALE

## Check truncation toward zero when scaling produces non-integer results
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -sched-model-reservation-station-scale-factor=0.7 -debug 2>&1 | FileCheck %s --check-prefix=DOWNSCALE-ROUND
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -sched-model-reservation-station-scale-factor=1.7 -debug 2>&1 | FileCheck %s --check-prefix=UPSCALE-ROUND

## Default scale factor is 1
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -debug 2>&1 | FileCheck %s --check-prefix=ORIGINAL

## Zero/negative scale factors are ignored
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -sched-model-reservation-station-scale-factor=0 -debug 2>&1 | FileCheck %s --check-prefix=ORIGINAL
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -sched-model-reservation-station-scale-factor=-1 -debug 2>&1 | FileCheck %s --check-prefix=ORIGINAL

## Scaling results <= 1 fall back to the original buffer size
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=apple-m1 -sched-model-reservation-station-scale-factor=0.01 -debug 2>&1 | FileCheck %s --check-prefix=ORIGINAL

## BufferSize=0 is not scaled
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=cortex-a53 -sched-model-reservation-station-scale-factor=2 -debug 2>&1 | FileCheck %s --check-prefix=BUFFER-ZERO

## BufferSize=1 is not scaled
# RUN: llvm-mca < %s -mtriple=aarch64 -mcpu=exynos-m5 -sched-model-reservation-station-scale-factor=2 -debug 2>&1 | FileCheck %s --check-prefix=BUFFER-ONE

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

# UPSCALE-FRAC-COUNT-1: Processor resources:
# UPSCALE-FRAC-NEXT: [ 0]  - 0x00000000000000 - InvalidUnit
# UPSCALE-FRAC-NEXT: [ 1]  - 0x00000000000001 - CyUnitB (BufferSize=36)
# UPSCALE-FRAC-NEXT: [ 2]  - 0x00000000000002 - CyUnitBR (BufferSize=-1)
# UPSCALE-FRAC-NEXT: [ 3]  - 0x00000000000004 - CyUnitFloatDiv (BufferSize=-1)
# UPSCALE-FRAC-NEXT: [ 4]  - 0x00000000000008 - CyUnitI (BufferSize=72)
# UPSCALE-FRAC-NEXT: [ 5]  - 0x00000000000010 - CyUnitID (BufferSize=24)
# UPSCALE-FRAC-NEXT: [ 6]  - 0x00000000000020 - CyUnitIM (BufferSize=48)
# UPSCALE-FRAC-NEXT: [ 7]  - 0x00000000000040 - CyUnitIS (BufferSize=36)
# UPSCALE-FRAC-NEXT: [ 8]  - 0x00000000000080 - CyUnitIntDiv (BufferSize=-1)
# UPSCALE-FRAC-NEXT: [ 9]  - 0x00000000000100 - CyUnitLS (BufferSize=42)
# UPSCALE-FRAC-NEXT: [10]  - 0x00000000000200 - CyUnitV (BufferSize=72)
# UPSCALE-FRAC-NEXT: [11]  - 0x00000000000400 - CyUnitVC (BufferSize=24)
# UPSCALE-FRAC-NEXT: [12]  - 0x00000000000800 - CyUnitVD (BufferSize=24)
# UPSCALE-FRAC-NEXT: [13]  - 0x00000000001000 - CyUnitVM (BufferSize=48)
# UPSCALE-FRAC:      [0] Code Region - foo

# DOWNSCALE-ROUND-COUNT-1: Processor resources:
# DOWNSCALE-ROUND-NEXT: [ 0]  - 0x00000000000000 - InvalidUnit
# DOWNSCALE-ROUND-NEXT: [ 1]  - 0x00000000000001 - CyUnitB (BufferSize=16)
# DOWNSCALE-ROUND-NEXT: [ 2]  - 0x00000000000002 - CyUnitBR (BufferSize=-1)
# DOWNSCALE-ROUND-NEXT: [ 3]  - 0x00000000000004 - CyUnitFloatDiv (BufferSize=-1)
# DOWNSCALE-ROUND-NEXT: [ 4]  - 0x00000000000008 - CyUnitI (BufferSize=33)
# DOWNSCALE-ROUND-NEXT: [ 5]  - 0x00000000000010 - CyUnitID (BufferSize=11)
# DOWNSCALE-ROUND-NEXT: [ 6]  - 0x00000000000020 - CyUnitIM (BufferSize=22)
# DOWNSCALE-ROUND-NEXT: [ 7]  - 0x00000000000040 - CyUnitIS (BufferSize=16)
# DOWNSCALE-ROUND-NEXT: [ 8]  - 0x00000000000080 - CyUnitIntDiv (BufferSize=-1)
# DOWNSCALE-ROUND-NEXT: [ 9]  - 0x00000000000100 - CyUnitLS (BufferSize=19)
# DOWNSCALE-ROUND-NEXT: [10]  - 0x00000000000200 - CyUnitV (BufferSize=33)
# DOWNSCALE-ROUND-NEXT: [11]  - 0x00000000000400 - CyUnitVC (BufferSize=11)
# DOWNSCALE-ROUND-NEXT: [12]  - 0x00000000000800 - CyUnitVD (BufferSize=11)
# DOWNSCALE-ROUND-NEXT: [13]  - 0x00000000001000 - CyUnitVM (BufferSize=22)
# DOWNSCALE-ROUND:      [0] Code Region - foo

# UPSCALE-ROUND-COUNT-1: Processor resources:
# UPSCALE-ROUND-NEXT: [ 0]  - 0x00000000000000 - InvalidUnit
# UPSCALE-ROUND-NEXT: [ 1]  - 0x00000000000001 - CyUnitB (BufferSize=40)
# UPSCALE-ROUND-NEXT: [ 2]  - 0x00000000000002 - CyUnitBR (BufferSize=-1)
# UPSCALE-ROUND-NEXT: [ 3]  - 0x00000000000004 - CyUnitFloatDiv (BufferSize=-1)
# UPSCALE-ROUND-NEXT: [ 4]  - 0x00000000000008 - CyUnitI (BufferSize=81)
# UPSCALE-ROUND-NEXT: [ 5]  - 0x00000000000010 - CyUnitID (BufferSize=27)
# UPSCALE-ROUND-NEXT: [ 6]  - 0x00000000000020 - CyUnitIM (BufferSize=54)
# UPSCALE-ROUND-NEXT: [ 7]  - 0x00000000000040 - CyUnitIS (BufferSize=40)
# UPSCALE-ROUND-NEXT: [ 8]  - 0x00000000000080 - CyUnitIntDiv (BufferSize=-1)
# UPSCALE-ROUND-NEXT: [ 9]  - 0x00000000000100 - CyUnitLS (BufferSize=47)
# UPSCALE-ROUND-NEXT: [10]  - 0x00000000000200 - CyUnitV (BufferSize=81)
# UPSCALE-ROUND-NEXT: [11]  - 0x00000000000400 - CyUnitVC (BufferSize=27)
# UPSCALE-ROUND-NEXT: [12]  - 0x00000000000800 - CyUnitVD (BufferSize=27)
# UPSCALE-ROUND-NEXT: [13]  - 0x00000000001000 - CyUnitVM (BufferSize=54)
# UPSCALE-ROUND:      [0] Code Region - foo

# BUFFER-ZERO-COUNT-1: Processor resources:
# BUFFER-ZERO-NEXT: [ 0]  - 0x00000000000000 - InvalidUnit
# BUFFER-ZERO-NEXT: [ 1]  - 0x00000000000001 - A53UnitALU (BufferSize=0)
# BUFFER-ZERO-NEXT: [ 2]  - 0x00000000000002 - A53UnitB (BufferSize=0)
# BUFFER-ZERO-NEXT: [ 3]  - 0x00000000000004 - A53UnitDiv (BufferSize=0)
# BUFFER-ZERO-NEXT: [ 4]  - 0x00000000000008 - A53UnitFPALU (BufferSize=0)
# BUFFER-ZERO-NEXT: [ 5]  - 0x00000000000010 - A53UnitFPMDS (BufferSize=0)
# BUFFER-ZERO-NEXT: [ 6]  - 0x00000000000020 - A53UnitLdSt (BufferSize=0)
# BUFFER-ZERO-NEXT: [ 7]  - 0x00000000000040 - A53UnitMAC (BufferSize=0)
# BUFFER-ZERO:      [0] Code Region - foo

# BUFFER-ONE-COUNT-1: Processor resources:
# BUFFER-ONE:        M5UnitD (BufferSize=1)
# BUFFER-ONE:        [0] Code Region - foo
