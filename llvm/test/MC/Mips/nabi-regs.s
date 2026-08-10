# OABI (o32, o64) have a different symbolic register
# set for the A and T registers because the NABI allows
# for 4 more register parameters (A registers) offsetting
# the T registers.
#
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding \
# RUN:   -mcpu=mips64r2 -arch=mips64 | FileCheck %s
#
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding \
# RUN:   -mcpu=mips64r2 -arch=mips64 -target-abi n32 | FileCheck %s
#
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding \
# RUN:   -mcpu=mips64r2 -arch=mips64 -target-abi n64 | FileCheck %s

    .text
foo:

# CHECK: add    $16, $16, $4            # encoding: [0x02,0x04,0x80,0x20]
    add $s0,$s0,$a0
# CHECK: add    $16, $16, $6            # encoding: [0x02,0x06,0x80,0x20]
    add $s0,$s0,$a2
# CHECK: add    $16, $16, $7            # encoding: [0x02,0x07,0x80,0x20]
    add $s0,$s0,$a3
# CHECK: add    $16, $16, $8            # encoding: [0x02,0x08,0x80,0x20]
    add $s0,$s0,$a4
# CHECK: add    $16, $16, $9            # encoding: [0x02,0x09,0x80,0x20]
    add $s0,$s0,$a5
# CHECK: add    $16, $16, $10           # encoding: [0x02,0x0a,0x80,0x20]
    add $s0,$s0,$a6
# CHECK: add    $16, $16, $11           # encoding: [0x02,0x0b,0x80,0x20]
    add $s0,$s0,$a7
# CHECK: add    $16, $16, $12           # encoding: [0x02,0x0c,0x80,0x20]
    add $s0,$s0,$t0
# CHECK: add    $16, $16, $13           # encoding: [0x02,0x0d,0x80,0x20]
    add $s0,$s0,$t1
# CHECK: add    $16, $16, $14           # encoding: [0x02,0x0e,0x80,0x20]
    add $s0,$s0,$t2
# CHECK: add    $16, $16, $15           # encoding: [0x02,0x0f,0x80,0x20]
    add $s0,$s0,$t3
