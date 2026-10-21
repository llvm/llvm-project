# RUN: not llvm-mc %s -triple=mips64 -mcpu=mips64r2 -target-abi=n32 2>&1 | \
# RUN:   FileCheck %s
# RUN: not llvm-mc %s -triple=mips64 -mcpu=mips3 -target-abi n64 2>&1 | \
# RUN:   FileCheck %s

  .text
  .option pic2
  dla $5, symbol+0x8000
  # CHECK: :[[@LINE-1]]:3: error: macro instruction uses large offset, which is not currently supported
  dla $5, symbol-0x8001
  # CHECK: :[[@LINE-1]]:3: error: macro instruction uses large offset, which is not currently supported
  dla $5, symbol+0x8000($6)
  # CHECK: :[[@LINE-1]]:3: error: macro instruction uses large offset, which is not currently supported
  dla $5, symbol-0x8001($6)
  # CHECK: :[[@LINE-1]]:3: error: macro instruction uses large offset, which is not currently supported
  dla $25, symbol+0x8000
  # CHECK: :[[@LINE-1]]:3: error: macro instruction uses large offset, which is not currently supported
  dla $25, symbol-0x8001
  # CHECK: :[[@LINE-1]]:3: error: macro instruction uses large offset, which is not currently supported
  dla $25, symbol+0x8000($6)
  # CHECK: :[[@LINE-1]]:3: error: macro instruction uses large offset, which is not currently supported
  dla $25, symbol-0x8001($6)
  # CHECK: :[[@LINE-1]]:3: error: macro instruction uses large offset, which is not currently supported

  dla $2, 24($f4)
  # CHECK: :[[@LINE-1]]:14: error: invalid operand for instruction
