# Instructions that are correctly rejected but emit a wrong or misleading error.
# RUN: not llvm-mc %s -triple=mips -show-encoding -mcpu=mips64r6 -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1


  # The LLD instruction with invalid memory operand should emit "expected memory with 12-bit signed offset".
  lld $31, 4096($31)       # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  lld $31, 2048($31)       # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  lld $31, -2049($31)      # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  # The LWU instruction with invalid memory operand should emit "expected memory with 12-bit signed offset".
  lwu $31, 4096($31)           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  lwu $31, 2048($31)           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  lwu $31, -2049($31)          # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  # The 10-bit immediate supported by the standard encodings cause us to emit
  # the diagnostic for the 10-bit form. This isn't exactly wrong but it is
  # misleading. Ideally, we'd emit every way to achieve a valid match instead
  # of picking only one.
  teq $8, $9, $2           # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
  teq $8, $9, -1           # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
  teq $8, $9, 16           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  tge $8, $9, $2           # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
  tge $8, $9, -1           # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
  tge $8, $9, 16           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  tgeu $8, $9, $2          # CHECK: :[[@LINE]]:16: error: expected 10-bit unsigned immediate
  tgeu $8, $9, -1          # CHECK: :[[@LINE]]:16: error: expected 10-bit unsigned immediate
  tgeu $8, $9, 16          # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  tlt $8, $9, $2           # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
  tlt $8, $9, -1           # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
  tlt $8, $9, 16           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  tltu $8, $9, $2          # CHECK: :[[@LINE]]:16: error: expected 10-bit unsigned immediate
  tltu $8, $9, -1          # CHECK: :[[@LINE]]:16: error: expected 10-bit unsigned immediate
  tltu $8, $9, 16          # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  tne $8, $9, $2           # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
  tne $8, $9, -1           # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
  tne $8, $9, 16           # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  dins $2, $3, -1, 1       # CHECK: :[[@LINE]]:16: error: expected 6-bit unsigned immediate
  syscall -1               # CHECK: :[[@LINE]]:11: error: expected 20-bit unsigned immediate
  syscall $4               # CHECK: :[[@LINE]]:11: error: expected 20-bit unsigned immediate
  syscall 1024             # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  ldc2 $1, -2049($12)      # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  ldc2 $1, 2048($12)       # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  ldc2 $1, 1023($32)       # CHECK: :[[@LINE]]:12: error: expected memory with 16-bit signed offset
  lwc2 $1, -2049($4)       # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  lwc2 $1, 2048($4)        # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  lwc2 $1, 16($32)         # CHECK: :[[@LINE]]:12: error: expected memory with 16-bit signed offset
  sdc2 $1, -2049($16)      # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  sdc2 $1, 2048($16)       # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  sdc2 $1, 8($32)          # CHECK: :[[@LINE]]:12: error: expected memory with 16-bit signed offset
  swc2 $1, -2049($17)      # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  swc2 $1, 2048($17)       # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  swc2 $1, 777($32)        # CHECK: :[[@LINE]]:12: error: expected memory with 16-bit signed offset
