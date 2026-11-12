# RUN: llvm-mc -triple mips-unknown-linux -show-encoding %s | FileCheck %s

.text
# CHECK:    .text
  abs $4, $4
# CHECK-NEXT: .set push
# CHECK-NEXT: .set noreorder
# CHECK-NEXT: bgez    $4, 8     # encoding: [0x04,0x81,0x00,0x02]
# CHECK-NEXT: nop               # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NEXT: .set pop
# CHECK-NEXT: neg     $4, $4    # encoding: [0x00,0x04,0x20,0x22]
  abs $4, $5
# CHECK-NEXT: .set push
# CHECK-NEXT: .set noreorder
# CHECK-NEXT: bgez    $5, 8     # encoding: [0x04,0xa1,0x00,0x02]
# CHECK-NEXT: move    $4, $5    # encoding: [0x00,0xa0,0x20,0x21]
# CHECK-NEXT: .set pop
# CHECK-NEXT: neg     $4, $5    # encoding: [0x00,0x05,0x20,0x22]
