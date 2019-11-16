# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r2 2>%t1
# RUN:   FileCheck %s < %t1 --check-prefix=O32
# RUN: not llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n32 2>&1 | \
# RUN:   FileCheck %s --check-prefix=N32
# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n64 2>&1 | \
# RUN:   FileCheck %s --check-prefix=N64

  .text
  la $5, 0x100000000
  # O32: :[[@LINE-1]]:3: error: instruction requires a 32-bit immediate
  # N32: :[[@LINE-2]]:3: error: instruction requires a 32-bit immediate
  # N64: :[[@LINE-3]]:3: warning: la used to load 64-bit address

  la $5, 0x100000000($6)
  # O32: :[[@LINE-1]]:3: error: instruction requires a 32-bit immediate
  # N32: :[[@LINE-2]]:3: error: instruction requires a 32-bit immediate
  # N64: :[[@LINE-3]]:3: warning: la used to load 64-bit address

  la $5, symbol
  # N32-NOT: :[[@LINE-1]]:3: warning: la used to load 64-bit address
  # N64:     :[[@LINE-2]]:3: warning: la used to load 64-bit address
