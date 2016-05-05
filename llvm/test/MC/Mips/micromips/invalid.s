# RUN: not llvm-mc %s -triple=mips -show-encoding -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1

  addiur1sp $7, 260   # CHECK: :[[@LINE]]:17: error: expected both 8-bit unsigned immediate and multiple of 4
  addiur1sp $7, 241   # CHECK: :[[@LINE]]:17: error: expected both 8-bit unsigned immediate and multiple of 4
  addiur1sp $8, 240   # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
  addius5 $2, -9      # CHECK: :[[@LINE]]:15: error: expected 4-bit signed immediate
  addius5 $2, 8       # CHECK: :[[@LINE]]:15: error: expected 4-bit signed immediate
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
  cachee 0, -513($7)  # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
  cachee 0, 512($7)   # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
  # FIXME: Check '0 < pos + size <= 32' constraint on ext
  ext $2, $3, -1, 31  # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ext $2, $3, 32, 31  # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ext $2, $3, 1, 0    # CHECK: :[[@LINE]]:18: error: expected immediate in range 1 .. 32
  ext $2, $3, 1, 33   # CHECK: :[[@LINE]]:18: error: expected immediate in range 1 .. 32
  ins $2, $3, -1, 31  # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ins $2, $3, 32, 31  # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ins $2, $3, -1, 1   # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ins $2, $3, 32, 1   # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ins $2, $3, 0, -1   # CHECK: :[[@LINE]]:18: error: expected immediate in range 1 .. 32
  ins $2, $3, 0, 33   # CHECK: :[[@LINE]]:18: error: expected immediate in range 1 .. 32
  jraddiusp -1        # CHECK: :[[@LINE]]:13: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp -4        # CHECK: :[[@LINE]]:13: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 125       # CHECK: :[[@LINE]]:13: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 128       # CHECK: :[[@LINE]]:13: error: expected both 7-bit unsigned immediate and multiple of 4
  li16 $4, -2         # CHECK: :[[@LINE]]:12: error: expected immediate in range -1 .. 126
  li16 $4, 127        # CHECK: :[[@LINE]]:12: error: expected immediate in range -1 .. 126
  pref -1, 255($7)    # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  pref 32, 255($7)    # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  prefe 0, -513($7)   # CHECK: :[[@LINE]]:12: error: expected memory with 9-bit signed offset
  prefe 0, 512($7)    # CHECK: :[[@LINE]]:12: error: expected memory with 9-bit signed offset
  rotr $2, -1         # CHECK: :[[@LINE]]:12: error: expected 5-bit unsigned immediate
  rotr $2, 32         # CHECK: :[[@LINE]]:12: error: expected 5-bit unsigned immediate
  rotr $2, $3, -1     # CHECK: :[[@LINE]]:16: error: expected 5-bit unsigned immediate
  rotr $2, $3, 32     # CHECK: :[[@LINE]]:16: error: expected 5-bit unsigned immediate
  rotrv $9, $6, 5     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sdbbp16 -1          # CHECK: :[[@LINE]]:11: error: expected 4-bit unsigned immediate
  sdbbp16 16          # CHECK: :[[@LINE]]:11: error: expected 4-bit unsigned immediate
  sll $2, $3, -1      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sll $2, $3, 32      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sra $2, $3, -1      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sra $2, $3, 32      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  srl $2, $3, -1      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  srl $2, $3, 32      # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sync -1             # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  sync 32             # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  swe $2, -513($gp)   # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  swe $2, 512($gp)    # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  swe $33, 8($gp)     # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  swe $2, 8($33)      # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sll $3, -1          # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sll $3, 32          # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sra $3, -1          # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sra $3, 32          # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  srl $3, -1          # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  lle $33, 8($5)      # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  lle $4, 8($33)      # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lle $4, 512($5)     # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lle $4, -513($5)    # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lwe $33, 8($5)      # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  lwe $4, 8($33)      # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lwe $4, 512($5)     # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lwe $4, -513($5)    # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sbe $33, 8($5)      # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  sbe $4, 8($33)      # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sbe $4, 512($5)     # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sbe $4, -513($5)    # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sce $33, 8($5)      # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  sce $4, 8($33)      # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sce $4, 512($5)     # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sce $4, -513($5)    # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  she $33, 8($5)      # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  she $4, 8($33)      # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  she $4, 512($5)     # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  she $4, -513($5)    # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
