# RUN: llvm-mc  %s -triple=mips -mcpu=mips64 -show-encoding | FileCheck %s -check-prefix=CHECK-64
# RUN: llvm-mc  %s -triple=mips -mcpu=mips64r2 -show-encoding | FileCheck %s -check-prefix=CHECK-64R
# RUN: llvm-mc  %s -triple=mips -mcpu=mips64r3 -show-encoding | FileCheck %s -check-prefix=CHECK-64R
# RUN: llvm-mc  %s -triple=mips -mcpu=mips64r5 -show-encoding | FileCheck %s -check-prefix=CHECK-64R
# RUN: llvm-mc  %s -triple=mips -mcpu=mips64r6 -show-encoding | FileCheck %s -check-prefix=CHECK-64R

  .text
foo:
  rol $4,$5
# CHECK-64:     subu    $1, $zero, $5       # encoding: [0x00,0x05,0x08,0x23]
# CHECK-64:     srlv    $1, $4, $1          # encoding: [0x00,0x24,0x08,0x06]
# CHECK-64:     sllv    $4, $4, $5          # encoding: [0x00,0xa4,0x20,0x04]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    subu    $1, $zero, $5       # encoding: [0x00,0x05,0x08,0x23]
# CHECK-64R:    rotrv   $4, $4, $1          # encoding: [0x00,0x24,0x20,0x46]
  rol $4,$5,$6
# CHECK-64:     subu    $1, $zero, $6       # encoding: [0x00,0x06,0x08,0x23]
# CHECK-64:     srlv    $1, $5, $1          # encoding: [0x00,0x25,0x08,0x06]
# CHECK-64:     sllv    $4, $5, $6          # encoding: [0x00,0xc5,0x20,0x04]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    negu    $4, $6              # encoding: [0x00,0x06,0x20,0x23]
# CHECK-64R:    rotrv   $4, $5, $4          # encoding: [0x00,0x85,0x20,0x46]
  rol $4,0
# CHECK-64:     srl     $4, $4, 0           # encoding: [0x00,0x04,0x20,0x02]
# CHECK-64R:    rotr    $4, $4, 0           # encoding: [0x00,0x24,0x20,0x02]
  rol $4,$5,0
# CHECK-64:     srl     $4, $5, 0           # encoding: [0x00,0x05,0x20,0x02]
# CHECK-64R:    rotr    $4, $5, 0           # encoding: [0x00,0x25,0x20,0x02]
  rol $4,1
# CHECK-64:     sll     $1, $4, 1           # encoding: [0x00,0x04,0x08,0x40]
# CHECK-64:     srl     $4, $4, 31          # encoding: [0x00,0x04,0x27,0xc2]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    rotr    $4, $4, 31          # encoding: [0x00,0x24,0x27,0xc2]
  rol $4,$5,1
# CHECK-64:     sll     $1, $5, 1           # encoding: [0x00,0x05,0x08,0x40]
# CHECK-64:     srl     $4, $5, 31          # encoding: [0x00,0x05,0x27,0xc2]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    rotr    $4, $5, 31          # encoding: [0x00,0x25,0x27,0xc2]
  rol $4,2
# CHECK-64:     sll     $1, $4, 2           # encoding: [0x00,0x04,0x08,0x80]
# CHECK-64:     srl     $4, $4, 30          # encoding: [0x00,0x04,0x27,0x82]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    rotr    $4, $4, 30          # encoding: [0x00,0x24,0x27,0x82]
  rol $4,$5,2
# CHECK-64:     sll     $1, $5, 2           # encoding: [0x00,0x05,0x08,0x80]
# CHECK-64:     srl     $4, $5, 30          # encoding: [0x00,0x05,0x27,0x82]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    rotr    $4, $5, 30          # encoding: [0x00,0x25,0x27,0x82]

  ror $4,$5
# CHECK-64:     subu    $1, $zero, $5       # encoding: [0x00,0x05,0x08,0x23]
# CHECK-64:     sllv    $1, $4, $1          # encoding: [0x00,0x24,0x08,0x04]
# CHECK-64:     srlv    $4, $4, $5          # encoding: [0x00,0xa4,0x20,0x06]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    rotrv   $4, $4, $5          # encoding: [0x00,0xa4,0x20,0x46]
  ror $4,$5,$6
# CHECK-64:     subu    $1, $zero, $6       # encoding: [0x00,0x06,0x08,0x23]
# CHECK-64:     sllv    $1, $5, $1          # encoding: [0x00,0x25,0x08,0x04]
# CHECK-64:     srlv    $4, $5, $6          # encoding: [0x00,0xc5,0x20,0x06]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    rotrv   $4, $5, $6          # encoding: [0x00,0xc5,0x20,0x46]
  ror $4,0
# CHECK-64:     srl     $4, $4, 0           # encoding: [0x00,0x04,0x20,0x02]
# CHECK-64R:    rotr    $4, $4, 0           # encoding: [0x00,0x24,0x20,0x02]
  ror $4,$5,0
# CHECK-64:     srl     $4, $5, 0           # encoding: [0x00,0x05,0x20,0x02]
# CHECK-64R:    rotr    $4, $5, 0           # encoding: [0x00,0x25,0x20,0x02]
  ror $4,1
# CHECK-64:     srl     $1, $4, 1           # encoding: [0x00,0x04,0x08,0x42]
# CHECK-64:     sll     $4, $4, 31          # encoding: [0x00,0x04,0x27,0xc0]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    rotr    $4, $4, 1           # encoding: [0x00,0x24,0x20,0x42]
  ror $4,$5,1
# CHECK-64:     srl     $1, $5, 1           # encoding: [0x00,0x05,0x08,0x42]
# CHECK-64:     sll     $4, $5, 31          # encoding: [0x00,0x05,0x27,0xc0]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    rotr    $4, $5, 1           # encoding: [0x00,0x25,0x20,0x42]
  ror $4,2
# CHECK-64:     srl     $1, $4, 2           # encoding: [0x00,0x04,0x08,0x82]
# CHECK-64:     sll     $4, $4, 30          # encoding: [0x00,0x04,0x27,0x80]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    rotr    $4, $4, 2           # encoding: [0x00,0x24,0x20,0x82]
  ror $4,$5,2
# CHECK-64:     srl     $1, $5, 2           # encoding: [0x00,0x05,0x08,0x82]
# CHECK-64:     sll     $4, $5, 30          # encoding: [0x00,0x05,0x27,0x80]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    rotr    $4, $5, 2           # encoding: [0x00,0x25,0x20,0x82]

  drol $4,$5
# CHECK-64:     dsubu   $1, $zero, $5       # encoding: [0x00,0x05,0x08,0x2f]
# CHECK-64:     dsrlv   $1, $4, $1          # encoding: [0x00,0x24,0x08,0x16]
# CHECK-64:     dsllv   $4, $4, $5          # encoding: [0x00,0xa4,0x20,0x14]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    dsubu   $1, $zero, $5       # encoding: [0x00,0x05,0x08,0x2f]
# CHECK-64R:    drotrv  $4, $4, $1          # encoding: [0x00,0x24,0x20,0x56]
  drol $4,$5,$6
# CHECK-64:     dsubu   $1, $zero, $6       # encoding: [0x00,0x06,0x08,0x2f]
# CHECK-64:     dsrlv   $1, $5, $1          # encoding: [0x00,0x25,0x08,0x16]
# CHECK-64:     dsllv   $4, $5, $6          # encoding: [0x00,0xc5,0x20,0x14]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    dsubu   $4, $zero, $6       # encoding: [0x00,0x06,0x20,0x2f]
# CHECK-64R:    drotrv  $4, $5, $4          # encoding: [0x00,0x85,0x20,0x56]

  drol $4,1
# CHECK-64:     dsll    $1, $4, 1           # encoding: [0x00,0x04,0x08,0x78]
# CHECK-64:     dsrl32  $4, $4, 31          # encoding: [0x00,0x04,0x27,0xfe]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $4, 31          # encoding: [0x00,0x24,0x27,0xfe]
  drol $4,$5,0
# CHECK-64:     dsrl    $4, $5, 0           # encoding: [0x00,0x05,0x20,0x3a]
# CHECK-64R:    drotr   $4, $5, 0           # encoding: [0x00,0x25,0x20,0x3a]
  drol $4,$5,1
# CHECK-64:     dsll    $1, $5, 1           # encoding: [0x00,0x05,0x08,0x78]
# CHECK-64:     dsrl32  $4, $5, 31          # encoding: [0x00,0x05,0x27,0xfe]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 31          # encoding: [0x00,0x25,0x27,0xfe]
  drol $4,$5,31
# CHECK-64:     dsll    $1, $5, 31          # encoding: [0x00,0x05,0x0f,0xf8]
# CHECK-64:     dsrl32  $4, $5, 1           # encoding: [0x00,0x05,0x20,0x7e]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 1           # encoding: [0x00,0x25,0x20,0x7e]
  drol $4,$5,32
# CHECK-64:     dsll32  $1, $5, 0           # encoding: [0x00,0x05,0x08,0x3c]
# CHECK-64:     dsrl32  $4, $5, 0           # encoding: [0x00,0x05,0x20,0x3e]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 0           # encoding: [0x00,0x25,0x20,0x3e]
  drol $4,$5,33
# CHECK-64:     dsll32  $1, $5, 1           # encoding: [0x00,0x05,0x08,0x7c]
# CHECK-64:     dsrl    $4, $5, 31          # encoding: [0x00,0x05,0x27,0xfa]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr   $4, $5, 31          # encoding: [0x00,0x25,0x27,0xfa]
  drol $4,$5,63
# CHECK-64:     dsll32  $1, $5, 31          # encoding: [0x00,0x05,0x0f,0xfc]
# CHECK-64:     dsrl    $4, $5, 1           # encoding: [0x00,0x05,0x20,0x7a]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr   $4, $5, 1           # encoding: [0x00,0x25,0x20,0x7a]
  drol $4,$5,64
# CHECK-64:     dsrl    $4, $5, 0           # encoding: [0x00,0x05,0x20,0x3a]
# CHECK-64R:    drotr   $4, $5, 0           # encoding: [0x00,0x25,0x20,0x3a]
  drol $4,$5,65
# CHECK-64:     dsll    $1, $5, 1           # encoding: [0x00,0x05,0x08,0x78]
# CHECK-64:     dsrl32  $4, $5, 31          # encoding: [0x00,0x05,0x27,0xfe]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 31          # encoding: [0x00,0x25,0x27,0xfe]
  drol $4,$5,95
# CHECK-64:     dsll    $1, $5, 31          # encoding: [0x00,0x05,0x0f,0xf8]
# CHECK-64:     dsrl32  $4, $5, 1           # encoding: [0x00,0x05,0x20,0x7e]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 1           # encoding: [0x00,0x25,0x20,0x7e]
  drol $4,$5,96
# CHECK-64:     dsll32  $1, $5, 0           # encoding: [0x00,0x05,0x08,0x3c]
# CHECK-64:     dsrl32  $4, $5, 0           # encoding: [0x00,0x05,0x20,0x3e]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 0           # encoding: [0x00,0x25,0x20,0x3e]
  drol $4,$5,97
# CHECK-64:     dsll32  $1, $5, 1           # encoding: [0x00,0x05,0x08,0x7c]
# CHECK-64:     dsrl    $4, $5, 31          # encoding: [0x00,0x05,0x27,0xfa]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr   $4, $5, 31          # encoding: [0x00,0x25,0x27,0xfa]
  drol $4,$5,127
# CHECK-64:     dsll32  $1, $5, 31          # encoding: [0x00,0x05,0x0f,0xfc]
# CHECK-64:     dsrl    $4, $5, 1           # encoding: [0x00,0x05,0x20,0x7a]
# CHECK-64:     or  $4, $4, $1              # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr   $4, $5, 1           # encoding: [0x00,0x25,0x20,0x7a]

  dror $4,$5
# CHECK-64:     dsubu   $1, $zero, $5       # encoding: [0x00,0x05,0x08,0x2f]
# CHECK-64:     dsllv   $1, $4, $1          # encoding: [0x00,0x24,0x08,0x14]
# CHECK-64:     dsrlv   $4, $4, $5          # encoding: [0x00,0xa4,0x20,0x16]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotrv  $4, $4, $5          # encoding: [0x00,0xa4,0x20,0x56]
  dror $4,$5,$6
# CHECK-64:     dsubu   $1, $zero, $6       # encoding: [0x00,0x06,0x08,0x2f]
# CHECK-64:     dsllv   $1, $5, $1          # encoding: [0x00,0x25,0x08,0x14]
# CHECK-64:     dsrlv   $4, $5, $6          # encoding: [0x00,0xc5,0x20,0x16]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotrv  $4, $5, $6          # encoding: [0x00,0xc5,0x20,0x56]
  dror $4,1
# CHECK-64:     dsrl    $1, $4, 1           # encoding: [0x00,0x04,0x08,0x7a]
# CHECK-64:     dsll32  $4, $4, 31          # encoding: [0x00,0x04,0x27,0xfc]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr   $4, $4, 1           # encoding: [0x00,0x24,0x20,0x7a]
  dror $4,$5,0
# CHECK-64:     dsrl    $4, $5, 0           # encoding: [0x00,0x05,0x20,0x3a]
# CHECK-64R:    drotr   $4, $5, 0           # encoding: [0x00,0x25,0x20,0x3a]
  dror $4,$5,1
# CHECK-64:     dsrl    $1, $5, 1           # encoding: [0x00,0x05,0x08,0x7a]
# CHECK-64:     dsll32  $4, $5, 31          # encoding: [0x00,0x05,0x27,0xfc]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr   $4, $5, 1           # encoding: [0x00,0x25,0x20,0x7a]
  dror $4,$5,31
# CHECK-64:     dsrl    $1, $5, 31          # encoding: [0x00,0x05,0x0f,0xfa]
# CHECK-64:     dsll32  $4, $5, 1           # encoding: [0x00,0x05,0x20,0x7c]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr   $4, $5, 31          # encoding: [0x00,0x25,0x27,0xfa]
  dror $4,$5,32
# CHECK-64:     dsrl32  $1, $5, 0           # encoding: [0x00,0x05,0x08,0x3e]
# CHECK-64:     dsll32  $4, $5, 0           # encoding: [0x00,0x05,0x20,0x3c]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 0           # encoding: [0x00,0x25,0x20,0x3e]
  dror $4,$5,33
# CHECK-64:     dsrl32  $1, $5, 1           # encoding: [0x00,0x05,0x08,0x7e]
# CHECK-64:     dsll    $4, $5, 31          # encoding: [0x00,0x05,0x27,0xf8]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 1           # encoding: [0x00,0x25,0x20,0x7e]
  dror $4,$5,63
# CHECK-64:     dsrl32  $1, $5, 31          # encoding: [0x00,0x05,0x0f,0xfe]
# CHECK-64:     dsll    $4, $5, 1           # encoding: [0x00,0x05,0x20,0x78]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 31          # encoding: [0x00,0x25,0x27,0xfe]
  dror $4,$5,64
# CHECK-64:     dsrl    $4, $5, 0           # encoding: [0x00,0x05,0x20,0x3a]
# CHECK-64R:    drotr   $4, $5, 0           # encoding: [0x00,0x25,0x20,0x3a]
  dror $4,$5,65
# CHECK-64:     dsrl    $1, $5, 1           # encoding: [0x00,0x05,0x08,0x7a]
# CHECK-64:     dsll32  $4, $5, 31          # encoding: [0x00,0x05,0x27,0xfc]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr   $4, $5, 1           # encoding: [0x00,0x25,0x20,0x7a]
  dror $4,$5,95
# CHECK-64:     dsrl    $1, $5, 31          # encoding: [0x00,0x05,0x0f,0xfa]
# CHECK-64:     dsll32  $4, $5, 1           # encoding: [0x00,0x05,0x20,0x7c]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr   $4, $5, 31          # encoding: [0x00,0x25,0x27,0xfa]
  dror $4,$5,96
# CHECK-64:     dsrl32  $1, $5, 0           # encoding: [0x00,0x05,0x08,0x3e]
# CHECK-64:     dsll32  $4, $5, 0           # encoding: [0x00,0x05,0x20,0x3c]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 0           # encoding: [0x00,0x25,0x20,0x3e]
  dror $4,$5,97
# CHECK-64:     dsrl32  $1, $5, 1           # encoding: [0x00,0x05,0x08,0x7e]
# CHECK-64:     dsll    $4, $5, 31          # encoding: [0x00,0x05,0x27,0xf8]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 1           # encoding: [0x00,0x25,0x20,0x7e]
  dror $4,$5,127
# CHECK-64:     dsrl32  $1, $5, 31          # encoding: [0x00,0x05,0x0f,0xfe]
# CHECK-64:     dsll    $4, $5, 1           # encoding: [0x00,0x05,0x20,0x78]
# CHECK-64:     or      $4, $4, $1          # encoding: [0x00,0x81,0x20,0x25]
# CHECK-64R:    drotr32 $4, $5, 31          # encoding: [0x00,0x25,0x27,0xfe]
