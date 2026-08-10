# RUN: llvm-mc  %s -triple=mips64-unknown-linux -show-encoding -mcpu=octeon+ \
# RUN:   | FileCheck %s

# CHECK: baddu $9, $6, $7             # encoding: [0x70,0xc7,0x48,0x28]
# CHECK: baddu $17, $18, $19          # encoding: [0x72,0x53,0x88,0x28]
# CHECK: baddu $2, $2, $3             # encoding: [0x70,0x43,0x10,0x28]
# CHECK: bbit0 $19, 22, foo           # encoding: [0xca,0x76,A,A]
# CHECK: bbit032 $fp, 11, foo         # encoding: [0xdb,0xcb,A,A]
# CHECK: bbit032 $8, 10, foo          # encoding: [0xd9,0x0a,A,A]
# CHECK: bbit1 $3, 31, foo            # encoding: [0xe8,0x7f,A,A]
# CHECK: bbit132 $24, 10, foo         # encoding: [0xfb,0x0a,A,A]
# CHECK: bbit132 $14, 14, foo         # encoding: [0xf9,0xce,A,A]
# CHECK: cins  $25, $10, 22, 2        # encoding: [0x71,0x59,0x15,0xb2]
# CHECK: cins  $9, $9, 17, 29         # encoding: [0x71,0x29,0xec,0x72]
# CHECK: cins32 $15, $2, 18, 8        # encoding: [0x70,0x4f,0x44,0xb3]
# CHECK: cins32 $22, $22, 9, 22       # encoding: [0x72,0xd6,0xb2,0x73]
# CHECK: cins32 $24, $ra, 0, 31       # encoding: [0x73,0xf8,0xf8,0x33]
# CHECK: cins32 $15, $15, 5, 5        # encoding: [0x71,0xef,0x29,0x73]
# CHECK: dmtc2 $2, 16455              # encoding: [0x48,0xa2,0x40,0x47]
# CHECK: dmfc2 $2, 64                 # encoding: [0x48,0x22,0x00,0x40]
# CHECK: dmul  $9, $6, $7             # encoding: [0x70,0xc7,0x48,0x03]
# CHECK: dmul  $19, $24, $25          # encoding: [0x73,0x19,0x98,0x03]
# CHECK: dmul  $9, $9, $6             # encoding: [0x71,0x26,0x48,0x03]
# CHECK: dmul  $21, $21, $25          # encoding: [0x72,0xb9,0xa8,0x03]
# CHECK: dpop  $9, $6                 # encoding: [0x70,0xc0,0x48,0x2d]
# CHECK: dpop  $15, $22               # encoding: [0x72,0xc0,0x78,0x2d]
# CHECK: dpop  $12, $12               # encoding: [0x71,0x80,0x60,0x2d]
# CHECK: exts  $4, $25, 27, 15        # encoding: [0x73,0x24,0x7e,0xfa]
# CHECK: exts  $15, $15, 17, 6        # encoding: [0x71,0xef,0x34,0x7a]
# CHECK: exts32 $4, $13, 10, 8        # encoding: [0x71,0xa4,0x42,0xbb]
# CHECK: exts32 $15, $15, 11, 20      # encoding: [0x71,0xef,0xa2,0xfb]
# CHECK: exts32 $7, $4, 22, 9         # encoding: [0x70,0x87,0x4d,0xbb]
# CHECK: exts32 $25, $25, 5, 25       # encoding: [0x73,0x39,0xc9,0x7b]
# CHECK: mtm0  $15                    # encoding: [0x71,0xe0,0x00,0x08]
# CHECK: mtm1  $16                    # encoding: [0x72,0x00,0x00,0x0c]
# CHECK: mtm2  $17                    # encoding: [0x72,0x20,0x00,0x0d]
# CHECK: mtp0  $18                    # encoding: [0x72,0x40,0x00,0x09]
# CHECK: mtp1  $19                    # encoding: [0x72,0x60,0x00,0x0a]
# CHECK: mtp2  $20                    # encoding: [0x72,0x80,0x00,0x0b]
# CHECK: pop   $9, $6                 # encoding: [0x70,0xc0,0x48,0x2c]
# CHECK: pop   $8, $19                # encoding: [0x72,0x60,0x40,0x2c]
# CHECK: pop   $2, $2                 # encoding: [0x70,0x40,0x10,0x2c]
# CHECK: saa   $2, ($5)               # encoding: [0x70,0xa2,0x00,0x18]
# CHECK: saad  $2, ($5)               # encoding: [0x70,0xa2,0x00,0x19]
# CHECK: seq   $25, $23, $24          # encoding: [0x72,0xf8,0xc8,0x2a]
# CHECK: seq   $6, $6, $24            # encoding: [0x70,0xd8,0x30,0x2a]
# CHECK: seqi  $17, $15, -512         # encoding: [0x71,0xf1,0x80,0x2e]
# CHECK: seqi  $16, $16, 38           # encoding: [0x72,0x10,0x09,0xae]
# CHECK: sne   $25, $23, $24          # encoding: [0x72,0xf8,0xc8,0x2b]
# CHECK: sne   $23, $23, $20          # encoding: [0x72,0xf4,0xb8,0x2b]
# CHECK: snei  $4, $16, -313          # encoding: [0x72,0x04,0xb1,0xef]
# CHECK: snei  $26, $26, 511          # encoding: [0x73,0x5a,0x7f,0xef]
# CHECK: sync  2                      # encoding: [0x00,0x00,0x00,0x8f]
# CHECK: sync  6                      # encoding: [0x00,0x00,0x01,0x8f]
# CHECK: sync  4                      # encoding: [0x00,0x00,0x01,0x0f]
# CHECK: sync  5                      # encoding: [0x00,0x00,0x01,0x4f]
# CHECK: v3mulu $21, $10, $21         # encoding: [0x71,0x55,0xa8,0x11]
# CHECK: v3mulu $20, $20, $10         # encoding: [0x72,0x8a,0xa0,0x11]
# CHECK: vmm0  $3, $19, $16           # encoding: [0x72,0x70,0x18,0x10]
# CHECK: vmm0  $ra, $ra, $9           # encoding: [0x73,0xe9,0xf8,0x10]
# CHECK: vmulu $sp, $10, $17          # encoding: [0x71,0x51,0xe8,0x0f]
# CHECK: vmulu $27, $27, $6           # encoding: [0x73,0x66,0xd8,0x0f]

foo:
  baddu   $9, $6, $7
  baddu   $17, $18, $19
  baddu   $2, $3
  bbit0   $19, 22, foo
  bbit032 $30, 11, foo
  bbit0   $8, 42, foo
  bbit1   $3, 31, foo
  bbit132 $24, 10, foo
  bbit1   $14, 46, foo
  cins    $25, $10, 22, 2
  cins    $9, 17, 29
  cins32  $15, $2, 18, 8
  cins32  $22, 9, 22
  cins    $24, $31, 32, 31
  cins    $15, 37, 5
  dmtc2   $2, 0x4047
  dmfc2   $2, 0x0040
  dmul    $9, $6, $7
  dmul    $19, $24, $25
  dmul    $9, $6
  dmul    $21, $25
  dpop    $9, $6
  dpop    $15, $22
  dpop    $12
  exts    $4, $25, 27, 15
  exts    $15, 17, 6
  exts32  $4, $13, 10, 8
  exts32  $15, 11, 20
  exts    $7, $4, 54, 9
  exts    $25, 37, 25
  mtm0    $15
  mtm1    $16
  mtm2    $17
  mtp0    $18
  mtp1    $19
  mtp2    $20
  pop     $9, $6
  pop     $8, $19
  pop     $2
  saa     $2, ($5)
  saad    $2, ($5)
  seq     $25, $23, $24
  seq     $6, $24
  seqi    $17, $15, -512
  seqi    $16, 38
  sne     $25, $23, $24
  sne     $23, $20
  snei    $4, $16, -313
  snei    $26, 511
  synciobdma
  syncs
  syncw
  syncws
  v3mulu  $21, $10, $21
  v3mulu  $20, $10
  vmm0    $3, $19, $16
  vmm0    $31, $9
  vmulu   $29, $10, $17
  vmulu   $27, $6
