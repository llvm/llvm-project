# RUN: llvm-mc  %s -triple=mips -mcpu=mips32 -show-encoding | FileCheck %s -check-prefix=CHECK-32
# RUN: llvm-mc  %s -triple=mips -mcpu=mips32r2 -show-encoding | FileCheck %s -check-prefix=CHECK-32R
# RUN: llvm-mc  %s -triple=mips -mcpu=mips32r3 -show-encoding | FileCheck %s -check-prefix=CHECK-32R
# RUN: llvm-mc  %s -triple=mips -mcpu=mips32r5 -show-encoding | FileCheck %s -check-prefix=CHECK-32R
# RUN: llvm-mc  %s -triple=mips -mcpu=mips32r6 -show-encoding | FileCheck %s -check-prefix=CHECK-32R

  .text
foo:
  rol $4,$5
# CHECK-32:     negu    $1, $5              # encoding: [0x00,0x05,0x08,0x23]
# CHECK-32:     srlv    $1, $4, $1          # encoding: [0x00,0x24,0x08,0x06]
# CHECK-32:     sllv    $4, $4, $5          # encoding: [0x00,0xa4,0x20,0x04]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    negu    $1, $5              # encoding: [0x00,0x05,0x08,0x23]
# CHECK-32R:    rotrv   $4, $4, $1          # encoding: [0x00,0x24,0x20,0x46]
  rol $4,$5,$6
# CHECK-32:     negu    $1, $6              # encoding: [0x00,0x06,0x08,0x23]
# CHECK-32:     srlv    $1, $5, $1          # encoding: [0x00,0x25,0x08,0x06]
# CHECK-32:     sllv    $4, $5, $6          # encoding: [0x00,0xc5,0x20,0x04]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    negu    $4, $6              # encoding: [0x00,0x06,0x20,0x23]
# CHECK-32R:    rotrv   $4, $5, $4          # encoding: [0x00,0x85,0x20,0x46]
  rol $4,0
# CHECK-32:     srl     $4, $4, 0           # encoding: [0x00,0x04,0x20,0x02]
# CHECK-32R:    rotr    $4, $4, 0           # encoding: [0x00,0x24,0x20,0x02]
  rol $4,$5,0
# CHECK-32:     srl     $4, $5, 0           # encoding: [0x00,0x05,0x20,0x02]
# CHECK-32R:    rotr    $4, $5, 0           # encoding: [0x00,0x25,0x20,0x02]
  rol $4,1
# CHECK-32:     sll     $1, $4, 1           # encoding: [0x00,0x04,0x08,0x40]
# CHECK-32:     srl     $4, $4, 31          # encoding: [0x00,0x04,0x27,0xc2]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    rotr    $4, $4, 31          # encoding: [0x00,0x24,0x27,0xc2]
  rol $4,$5,1
# CHECK-32:     sll     $1, $5, 1           # encoding: [0x00,0x05,0x08,0x40]
# CHECK-32:     srl     $4, $5, 31          # encoding: [0x00,0x05,0x27,0xc2]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    rotr    $4, $5, 31          # encoding: [0x00,0x25,0x27,0xc2]
  rol $4,2
# CHECK-32:     sll     $1, $4, 2           # encoding: [0x00,0x04,0x08,0x80]
# CHECK-32:     srl     $4, $4, 30          # encoding: [0x00,0x04,0x27,0x82]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    rotr    $4, $4, 30          # encoding: [0x00,0x24,0x27,0x82]
  rol $4,$5,2
# CHECK-32:     sll     $1, $5, 2           # encoding: [0x00,0x05,0x08,0x80]
# CHECK-32:     srl     $4, $5, 30          # encoding: [0x00,0x05,0x27,0x82]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    rotr    $4, $5, 30          # encoding: [0x00,0x25,0x27,0x82]

  ror $4,$5
# CHECK-32:     negu    $1, $5              # encoding: [0x00,0x05,0x08,0x23]
# CHECK-32:     sllv    $1, $4, $1          # encoding: [0x00,0x24,0x08,0x04]
# CHECK-32:     srlv    $4, $4, $5          # encoding: [0x00,0xa4,0x20,0x06]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    rotrv   $4, $4, $5          # encoding: [0x00,0xa4,0x20,0x46]
  ror $4,$5,$6
# CHECK-32:     negu    $1, $6              # encoding: [0x00,0x06,0x08,0x23]
# CHECK-32:     sllv    $1, $5, $1          # encoding: [0x00,0x25,0x08,0x04]
# CHECK-32:     srlv    $4, $5, $6          # encoding: [0x00,0xc5,0x20,0x06]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    rotrv   $4, $5, $6          # encoding: [0x00,0xc5,0x20,0x46]
  ror $4,0
# CHECK-32:     srl     $4, $4, 0           # encoding: [0x00,0x04,0x20,0x02]
# CHECK-32R:    rotr    $4, $4, 0           # encoding: [0x00,0x24,0x20,0x02]
  ror $4,$5,0
# CHECK-32:     srl     $4, $5, 0           # encoding: [0x00,0x05,0x20,0x02]
# CHECK-32R:    rotr    $4, $5, 0           # encoding: [0x00,0x25,0x20,0x02]
  ror $4,1
# CHECK-32:     srl     $1, $4, 1           # encoding: [0x00,0x04,0x08,0x42]
# CHECK-32:     sll     $4, $4, 31          # encoding: [0x00,0x04,0x27,0xc0]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    rotr    $4, $4, 1           # encoding: [0x00,0x24,0x20,0x42]
  ror $4,$5,1
# CHECK-32:     srl     $1, $5, 1           # encoding: [0x00,0x05,0x08,0x42]
# CHECK-32:     sll     $4, $5, 31          # encoding: [0x00,0x05,0x27,0xc0]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    rotr    $4, $5, 1           # encoding: [0x00,0x25,0x20,0x42]
  ror $4,2
# CHECK-32:     srl     $1, $4, 2           # encoding: [0x00,0x04,0x08,0x82]
# CHECK-32:     sll     $4, $4, 30          # encoding: [0x00,0x04,0x27,0x80]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    rotr    $4, $4, 2           # encoding: [0x00,0x24,0x20,0x82]
  ror $4,$5,2
# CHECK-32:     srl     $1, $5, 2           # encoding: [0x00,0x05,0x08,0x82]
# CHECK-32:     sll     $4, $5, 30          # encoding: [0x00,0x05,0x27,0x80]
# CHECK-32:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-32R:    rotr    $4, $5, 2           # encoding: [0x00,0x25,0x20,0x82]
