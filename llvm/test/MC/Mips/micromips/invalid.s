# RUN: not llvm-mc %s -triple=mips -show-encoding -mattr=micromips,eva 2>%t1
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
  swe $33, 8($gp)     # CHECK: :[[@LINE]]:7: error: invalid register number
  swe $2, 8($33)      # CHECK: :[[@LINE]]:13: error: invalid register number
  sll $3, -1          # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sll $3, 32          # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sra $3, -1          # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sra $3, 32          # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  srl $3, -1          # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  lle $33, 8($5)      # CHECK: :[[@LINE]]:7: error: invalid register number
  lle $4, 8($33)      # CHECK: :[[@LINE]]:13: error: invalid register number
  lle $4, 512($5)     # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lle $4, -513($5)    # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lwe $33, 8($5)      # CHECK: :[[@LINE]]:7: error: invalid register number
  lwe $4, 8($33)      # CHECK: :[[@LINE]]:13: error: invalid register number
  lwe $4, 512($5)     # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lwe $4, -513($5)    # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sbe $33, 8($5)      # CHECK: :[[@LINE]]:7: error: invalid register number
  sbe $4, 8($33)      # CHECK: :[[@LINE]]:13: error: invalid register number
  sbe $4, 512($5)     # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sbe $4, -513($5)    # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sce $33, 8($5)      # CHECK: :[[@LINE]]:7: error: invalid register number
  sce $4, 8($33)      # CHECK: :[[@LINE]]:13: error: invalid register number
  sce $4, 512($5)     # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sce $4, -513($5)    # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  she $33, 8($5)      # CHECK: :[[@LINE]]:7: error: invalid register number
  she $4, 8($33)      # CHECK: :[[@LINE]]:13: error: invalid register number
  she $4, 512($5)     # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  she $4, -513($5)    # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lh $33, 8($4)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lhe $34, 8($2)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lhu $35, 8($2)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lhue $36, 8($2)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lh $2, 8($34)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lhe $4, 8($33)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lhu $4, 8($35)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lhue $4, 8($37)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lh $2, -65536($4)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  lh $2, 65536($4)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  lhe $4, -512($2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lhe $4, 512($2)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lhu $4, -65536($2)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  lhu $4, 65536($2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  lhue $4, -512($2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lhue $4, 512($2)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lwp $31, 8($4)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
                      # FIXME: This ought to point at the $34 but memory is treated as one operand.
  lwp $16, 8($34)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lwp $16, 4096($4)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 12-bit signed offset
  lwp $16, 8($16)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
  swp $31, 8($4)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  swp $16, 8($34)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  swp $16, 4096($4)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 12-bit signed offset
  andi $3, $4, -1     # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  andi $3, $4, 65536  # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  andi $3, -1         # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  andi $3, 65536      # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  ori $3, $4, -1      # CHECK: :[[@LINE]]:15: error: expected 16-bit unsigned immediate
  ori $3, $4, 65536   # CHECK: :[[@LINE]]:15: error: expected 16-bit unsigned immediate
  ori $3, -1          # CHECK: :[[@LINE]]:11: error: expected 16-bit unsigned immediate
  ori $3, 65536       # CHECK: :[[@LINE]]:11: error: expected 16-bit unsigned immediate
  xori $3, $4, -1     # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  xori $3, $4, 65536  # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  xori $3, -1         # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  xori $3, 65536      # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  not $3, 4           # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  lb $32, 8($5)       # CHECK: :[[@LINE]]:6: error: invalid register number
  lb $4, -32769($5)   # CHECK: :[[@LINE]]:10: error: expected memory with 16-bit signed offset
  lb $4, 32768($5)    # CHECK: :[[@LINE]]:10: error: expected memory with 16-bit signed offset
  lb $4, 8($32)       # CHECK: :[[@LINE]]:12: error: invalid register number
  lbu $32, 8($5)      # CHECK: :[[@LINE]]:7: error: invalid register number
  lbu $4, -32769($5)  # CHECK: :[[@LINE]]:11: error: expected memory with 16-bit signed offset
  lbu $4, 32768($5)   # CHECK: :[[@LINE]]:11: error: expected memory with 16-bit signed offset
  lbu $4, 8($32)      # CHECK: :[[@LINE]]:13: error: invalid register number
