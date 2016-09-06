# RUN: llvm-mc -show-encoding -mcpu=mips32 -triple mips-unknown-unknown %s | FileCheck %s
#
# CHECK:  .text
# CHECK:  $BB0_2:
# CHECK:  .abicalls
$BB0_2:
  .ent directives_test
     .abicalls
    .frame    $sp,0,$ra
    .mask     0x00000000,0
    .fmask    0x00000000,0

# CHECK: .set noreorder
# CHECK:   b 1332               # encoding: [0x10,0x00,0x01,0x4d]
# CHECK-NOT: nop
# CHECK:   j 1328               # encoding: [0x08,0x00,0x01,0x4c]
# CHECK-NOT: nop
# CHECK:   jal 1328             # encoding: [0x0c,0x00,0x01,0x4c]
# CHECK-NOT: nop
# CHECK: .set nomacro

    .set    noreorder
     b 1332
     j 1328
     jal 1328
    .set    nomacro
    .set    noat
$JTI0_0:
    .gpword    ($BB0_2)

    .word 0x77fffffc
# CHECK: $JTI0_0:
# CHECK: .gpword ($BB0_2)
# CHECK:     .4byte    2013265916
    .set  at=$12
    .set macro
# CHECK:   .set macro
# CHECK:   .set reorder
# CHECK:   b 1332               # encoding: [0x10,0x00,0x01,0x4d]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   j 1328               # encoding: [0x08,0x00,0x01,0x4c]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jal 1328             # encoding: [0x0c,0x00,0x01,0x4c]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
    .set reorder
$BB0_4:
    b 1332
    j 1328
    jal 1328
    .set  at=$a0
    .set STORE_MASK,$t7
    .set FPU_MASK,$f7
    .set  $tmp7, $BB0_4-$BB0_2
    .set f6,$f6
# CHECK:    abs.s   $f6, $f7           # encoding: [0x46,0x00,0x39,0x85]
# CHECK:    lui     $1, %hi($tmp7)     # encoding: [0x3c,0x01,A,A]
# CHECK:                               #   fixup A - offset: 0, value: %hi($tmp7), kind: fixup_Mips_HI16
    abs.s  f6,FPU_MASK
    lui $1, %hi($tmp7)

# CHECK:    .set mips32r2
# CHECK:    ldxc1   $f0, $zero($5)     # encoding: [0x4c,0xa0,0x00,0x01]
# CHECK:    luxc1   $f0, $6($5)        # encoding: [0x4c,0xa6,0x00,0x05]
# CHECK:    lwxc1   $f6, $2($5)        # encoding: [0x4c,0xa2,0x01,0x80]
     .set mips32r2
    ldxc1   $f0, $zero($5)
    luxc1   $f0, $6($5)
    lwxc1   $f6, $2($5)

# CHECK: .set mips64
# CHECK: dadd $3, $3, $3
    .set mips64
    dadd   $3, $3, $3                  # encoding: [0x00,0x62,0x18,0x2c]

# CHECK: .set mips64r2
# CHECK: drotr $9, $6, 30              # encoding: [0x00,0x26,0x4f,0xba]
    .set mips64r2
    drotr   $9, $6, 30

# CHECK:   .set dsp
# CHECK:   lbux    $7, $10($11)         # encoding: [0x7d,0x6a,0x39,0x8a]
# CHECK:   lhx     $5, $6($7)           # encoding: [0x7c,0xe6,0x29,0x0a]
   .set dsp
   lbux    $7, $10($11)
   lhx     $5, $6($7)
