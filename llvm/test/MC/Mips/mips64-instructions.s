# RUN: llvm-mc  %s -triple=mips64el-unknown-linux -show-encoding -mcpu=mips64r2 | FileCheck %s

# CHECK: ldxc1 $f2, $2($10)           # encoding: [0x81,0x00,0x42,0x4d]
# CHECK: sdxc1 $f8, $4($25)           # encoding: [0x09,0x40,0x24,0x4f]

  ldxc1 $f2, $2($10)
  sdxc1 $f8, $a0($t9)
