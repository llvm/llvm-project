# RUN: not llvm-mc %s -triple=mips -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1 --check-prefix=32-BIT
# RUN: not llvm-mc %s -triple=mips64 -mcpu=mips64 -target-abi n32 2>&1 | \
# RUN:   FileCheck %s --check-prefix=64-BIT
# RUN: not llvm-mc %s -triple=mips64 -mcpu=mips64 -target-abi n64 2>&1 | \
# RUN:   FileCheck %s --check-prefix=64-BIT

  .text
  li $5, 0x100000000
  # 32-BIT: :[[@LINE-1]]:10: error: expected 32-bit immediate
  # 64-BIT: :[[@LINE-2]]:10: error: expected 32-bit immediate
