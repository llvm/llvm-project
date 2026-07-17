# RUN: llvm-mc %s -triple=nanomips -show-encoding -show-inst \
# RUN:   -mattr=+crc | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple nanomips-elf -mattr=+crc %s -o - \
# RUN:   | llvm-objdump --triple nanomips-elf --mattr=+crc -dr - | FileCheck --check-prefixes DISAS %s

  .set noat
  crc32b $at, $t0       # CHECK: crc32b $at, $t0   # encoding: [0x2c,0x20,0xe8,0x03]
                        # CHECK-NEXT: # <MCInst #{{.*}} CRC32B_NM
                        # DISAS: {{.*}}: 2c 20 e8 03     crc32b  $at, $t0
  crc32h $a0, $a1       # CHECK: crc32h $a0, $a1   # encoding: [0x85,0x20,0xe8,0x07]
                        # CHECK-NEXT: # <MCInst #{{.*}} CRC32H_NM
                        # DISAS: {{.*}}: 85 20 e8 07     crc32h  $a0, $a1
  crc32w $a3, $a4       # CHECK: crc32w $a3, $a4   # encoding: [0xe8,0x20,0xe8,0x0b]
                        # CHECK-NEXT: # <MCInst #{{.*}} CRC32W_NM
                        # DISAS: {{.*}}: e8 20 e8 0b     crc32w  $a3, $a4
  crc32cb $at, $s0      # CHECK: crc32cb $at, $s0  # encoding: [0x30,0x20,0xe8,0x13]
                        # CHECK-NEXT: # <MCInst #{{.*}} CRC32CB_NM
                        # DISAS: {{.*}}: 30 20 e8 13     crc32cb  $at, $s0
  crc32ch $a0, $a1      # CHECK: crc32ch $a0, $a1  # encoding: [0x85,0x20,0xe8,0x17]
                        # CHECK-NEXT: # <MCInst #{{.*}} CRC32CH_NM
                        # DISAS: {{.*}}: 85 20 e8 17     crc32ch  $a0, $a1
  crc32cw $a3, $a4      # CHECK: crc32cw $a3, $a4  # encoding: [0xe8,0x20,0xe8,0x1b]
                        # CHECK-NEXT: # <MCInst #{{.*}} CRC32CW_NM
                        # DISAS: {{.*}}: e8 20 e8 1b     crc32cw  $a3, $a4
