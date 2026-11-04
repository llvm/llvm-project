# cfc2 and ctc2 outside microMIPS. The encodings match GNU as 2.47.

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

  cfc2 $2, $1
  ctc2 $8, $1
  cfc2 $31, $31
  ctc2 $0, $2

# CHECK: cfc2 $2, $1      # encoding: [0x48,0x42,0x08,0x00]
# CHECK: ctc2 $8, $1      # encoding: [0x48,0xc8,0x08,0x00]
# CHECK: cfc2 $ra, $31    # encoding: [0x48,0x5f,0xf8,0x00]
# CHECK: ctc2 $zero, $2   # encoding: [0x48,0xc0,0x10,0x00]
