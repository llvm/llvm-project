# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1 --check-prefix=32-BIT
# RUN: not llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n32 2>&1 | \
# RUN:   FileCheck %s --check-prefixes=64-BIT,N32-ONLY
# RUN: not llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n64 2>&1 | \
# RUN:   FileCheck %s --check-prefixes=64-BIT,N64-ONLY

  .text
  dli $5, 1
  # 32-BIT: :[[@LINE-1]]:3: error: instruction requires a 64-bit architecture
  bne $2, 0x100010001, 1332
  # 32-BIT: :[[@LINE-1]]:3: error: instruction requires a 32-bit immediate
  beq $2, 0x100010001, 1332
  # 32-BIT: :[[@LINE-1]]:3: error: instruction requires a 32-bit immediate
  .set mips32r6
  ulh $5, 0
  # 32-BIT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6
  ulhu $5, 0
  # 32-BIT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6
  .set mips32
  ulh $5, 1
  # 32-BIT-NOT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT-NOT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6
  ulhu $5, 1
  # 32-BIT-NOT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT-NOT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6
  .set mips64r6
  ulh $5, 2
  # 32-BIT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6
  ulhu $5, 2
  # 32-BIT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6

  .set mips32r6
  ulw $5, 0
  # 32-BIT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6
  .set mips32
  ulw $5, 1
  # 32-BIT-NOT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT-NOT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6
  .set mips64r6
  ulw $5, 2
  # 32-BIT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6
