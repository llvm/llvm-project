# RUN: llvm-mc -triple mips-mti-linux-gnu -show-encoding %s | FileCheck \
# RUN:         --check-prefixes=ALL,32 %s
# RUN: llvm-mc -triple mips64-mti-linux-gnu -show-encoding %s -target-abi n64 \
# RUN:         | FileCheck --check-prefixes=ALL,64 %s
# RUN: llvm-mc -triple mips64-mti-linux-gnu -show-encoding %s -target-abi n32 \
# RUN:         | FileCheck --check-prefixes=ALL,64 %s

# ALL:  .text
  ld $8, 0($5)
# 32:   lw  $8, 0($5)      # encoding: [0x8c,0xa8,0x00,0x00]
# 32:   lw  $9, 4($5)      # encoding: [0x8c,0xa9,0x00,0x04]
# 64:   ld  $8, 0($5)      # encoding: [0xdc,0xa8,0x00,0x00]
  sd $8, 0($5)
# 32:   sw  $8, 0($5)      # encoding: [0xac,0xa8,0x00,0x00]
# 32:   sw  $9, 4($5)      # encoding: [0xac,0xa9,0x00,0x04]
# 64:   sd  $8, 0($5)      # encoding: [0xfc,0xa8,0x00,0x00]
  ld $8, 0($8)
# 32:   lw  $9, 4($8)      # encoding: [0x8d,0x09,0x00,0x04]
# 32:   lw  $8, 0($8)      # encoding: [0x8d,0x08,0x00,0x00]
# 64:   ld  $8, 0($8)      # encoding: [0xdd,0x08,0x00,0x00]
  sd $8, 0($8)
# 32:   sw  $8, 0($8)      # encoding: [0xad,0x08,0x00,0x00]
# 32:   sw  $9, 4($8)      # encoding: [0xad,0x09,0x00,0x04]
# 64:   sd  $8, 0($8)      # encoding: [0xfd,0x08,0x00,0x00]
