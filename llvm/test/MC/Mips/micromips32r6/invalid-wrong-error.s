# Instructions that are correctly rejected but emit a wrong or misleading error.
# RUN: not llvm-mc %s -triple=mips -show-encoding -mcpu=mips32r6 -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1


  # The 10-bit immediate supported by the standard encodings cause us to emit
  # the diagnostic for the 10-bit form. This isn't exactly wrong but it is
  # misleading. Ideally, we'd emit every way to achieve a valid match instead
  # of picking only one.
  teq $8, $9, 16           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  tge $8, $9, 16           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  tgeu $8, $9, 16          # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  tlt $8, $9, 16           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  tltu $8, $9, 16          # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  tne $8, $9, 16           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  syscall 1024             # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  ldc2 $1, -2049($12)      # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  ldc2 $1, 2048($12)       # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  lwc2 $1, -2049($4)       # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  lwc2 $1, 2048($4)        # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  sdc2 $1, -2049($16)      # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  sdc2 $1, 2048($16)       # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  swc2 $1, -2049($17)      # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  swc2 $1, 2048($17)       # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  lwc2 $11, -1025($12)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  lwc2 $11, 1024($12)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  sc $4, 512($5)           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  sc $4, -513($5)          # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  ll $4, 512($5)           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  ll $4, -513($5)          # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
