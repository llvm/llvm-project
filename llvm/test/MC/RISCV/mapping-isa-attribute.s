## Test that .attribute 5 (Tag_RISCV_arch) updates the pending ISA mapping
## symbol.  Unlike .option arch, .attribute 5 is usually placed at the top
## of the file before any instruction is emitted, so it should *replace*
## (not add to) the initial "$x<base-ISA>" symbol scheduled by the target
## streamer constructor.

# RUN: llvm-mc -triple=riscv64 -filetype=obj -o %t.o %s
# RUN: llvm-readobj --symbols %t.o | FileCheck %s

.attribute 5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.text
## The initial mapping symbol reflects the arch widened by .attribute, not
## the triple-derived base.
vsetvli a3, a2, e8, m8, tu, mu
# CHECK: Name: $xrv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0
# CHECK-NEXT: Value: 0x0
