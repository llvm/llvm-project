# Instructions that are correctly rejected but emit a wrong or misleading error.
# RUN: not llvm-mc %s -triple=mips -show-encoding -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1

  # The 20-bit immediate supported by the standard encodings cause us to emit
  # the diagnostic for the 20-bit form. This isn't exactly wrong but it is
  # misleading. Ideally, we'd emit every way to achieve a valid match instead
  # of picking only one.
  sdbbp -1            # CHECK: :[[@LINE]]:9: error: expected 20-bit unsigned immediate
  sdbbp 1024          # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
  syscall -1          # CHECK: :[[@LINE]]:11: error: expected 20-bit unsigned immediate
  syscall $4          # CHECK: :[[@LINE]]:11: error: expected 20-bit unsigned immediate
  syscall 1024        # CHECK: :[[@LINE]]:3: error: instruction requires a CPU feature not currently enabled
