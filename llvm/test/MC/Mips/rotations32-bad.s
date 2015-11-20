# RUN: not llvm-mc  %s -arch=mips -mcpu=mips32 -show-encoding 2> %t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc  %s -arch=mips -mcpu=mips32r2 -show-encoding 2> %t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc  %s -arch=mips -mcpu=mips32r3 -show-encoding 2> %t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc  %s -arch=mips -mcpu=mips32r5 -show-encoding 2> %t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc  %s -arch=mips -mcpu=mips32r6 -show-encoding 2> %t1
# RUN: FileCheck %s < %t1

  .text
foo:

  drol $4,$5
# CHECK:        [[@LINE-1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  drol $4,$5,$6
# CHECK:        [[@LINE-1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  drol $4,0
# CHECK:        [[@LINE-1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  drol $4,$5,0
# CHECK:        [[@LINE-1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled

  dror $4,$5
# CHECK:        [[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  dror $4,$5,$6
# CHECK:        [[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  dror $4,0
# CHECK:        [[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  dror $4,$5,0
# CHECK:        [[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
