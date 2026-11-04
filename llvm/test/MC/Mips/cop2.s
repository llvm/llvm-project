# cop2 takes a 25-bit coprocessor operation. The encodings match GNU as 2.47.

# RUN: llvm-mc %s -triple=mips-unknown-linux-gnu -mcpu=mips1 -show-encoding \
# RUN:   | FileCheck %s
# RUN: llvm-mc %s -triple=mips64-unknown-linux-gnu -mcpu=mips3 -show-encoding \
# RUN:   | FileCheck %s
# RUN: llvm-mc %s -triple=mips64-unknown-linux-gnu -mcpu=mips3 -target-abi n32 \
# RUN:   -show-encoding | FileCheck %s
# RUN: llvm-mc %s -triple=mips64-unknown-linux-gnu -mcpu=mips64 -show-encoding \
# RUN:   | FileCheck %s
# RUN: llvm-mc %s -triple=mips-unknown-linux-gnu -mcpu=mips32r6 -show-encoding \
# RUN:   | FileCheck %s

  cop2 0
  cop2 0x123456
  cop2 0x1ffffff
  .set vd, 1
  cop2 (vd << 6)

# CHECK: cop2 0           # encoding: [0x4a,0x00,0x00,0x00]
# CHECK: cop2 1193046     # encoding: [0x4a,0x12,0x34,0x56]
# CHECK: cop2 33554431    # encoding: [0x4b,0xff,0xff,0xff]
# CHECK: cop2 64          # encoding: [0x4a,0x00,0x00,0x40]
