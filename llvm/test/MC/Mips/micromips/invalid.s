# RUN: not llvm-mc %s -triple=mips -show-encoding -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1

  break -1            # CHECK: :[[@LINE]]:9: error: expected 10-bit unsigned immediate
  break 1024          # CHECK: :[[@LINE]]:9: error: expected 10-bit unsigned immediate
  break -1, 5         # CHECK: :[[@LINE]]:9: error: expected 10-bit unsigned immediate
  break 1024, 5       # CHECK: :[[@LINE]]:9: error: expected 10-bit unsigned immediate
  break 7, -1         # CHECK: :[[@LINE]]:12: error: expected 10-bit unsigned immediate
  break 7, 1024       # CHECK: :[[@LINE]]:12: error: expected 10-bit unsigned immediate
  break16 -1          # CHECK: :[[@LINE]]:11: error: expected 4-bit unsigned immediate
  break16 16          # CHECK: :[[@LINE]]:11: error: expected 4-bit unsigned immediate
  cache -1, 255($7)   # CHECK: :[[@LINE]]:9: error: expected 5-bit unsigned immediate
  cache 32, 255($7)   # CHECK: :[[@LINE]]:9: error: expected 5-bit unsigned immediate
  # FIXME: Check '0 < pos + size <= 32' constraint on ext
  ext $2, $3, -1, 31  # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ext $2, $3, 32, 31  # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ext $2, $3, 1, 0    # CHECK: :[[@LINE]]:18: error: expected immediate in range 1 .. 32
  ext $2, $3, 1, 33   # CHECK: :[[@LINE]]:18: error: expected immediate in range 1 .. 32
  ins $2, $3, -1, 31  # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ins $2, $3, 32, 31  # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  jraddiusp -1        # CHECK: :[[@LINE]]:13: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp -4        # CHECK: :[[@LINE]]:13: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 125       # CHECK: :[[@LINE]]:13: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 128       # CHECK: :[[@LINE]]:13: error: expected both 7-bit unsigned immediate and multiple of 4
  pref -1, 255($7)    # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  pref 32, 255($7)    # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  rotr $2, $3, 32     # CHECK: :[[@LINE]]:16: error: expected 5-bit unsigned immediate
  sdbbp16 -1          # CHECK: :[[@LINE]]:11: error: expected 4-bit unsigned immediate
  sdbbp16 16          # CHECK: :[[@LINE]]:11: error: expected 4-bit unsigned immediate
  sll $2, $3, -1      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sll $2, $3, 32      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sra $2, $3, -1      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sra $2, $3, 32      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  srl $2, $3, -1      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  srl $2, $3, 32      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
