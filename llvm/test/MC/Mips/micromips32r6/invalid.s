# RUN: not llvm-mc %s -triple=mips -show-encoding -mcpu=mips32r6 -mattr=+micromips,+eva 2>%t1
# RUN: FileCheck %s < %t1

  addiur1sp $7, 260        # CHECK: :[[@LINE]]:17: error: expected both 8-bit unsigned immediate and multiple of 4
  addiur1sp $7, 241        # CHECK: :[[@LINE]]:17: error: expected both 8-bit unsigned immediate and multiple of 4
  addiur1sp $8, 240        # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
  addiur2 $9, $7, -1       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  addiur2 $6, $7, 10       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addius5 $2, -9           # CHECK: :[[@LINE]]:15: error: expected 4-bit signed immediate
  addius5 $2, 8            # CHECK: :[[@LINE]]:15: error: expected 4-bit signed immediate
  addiusp 1032             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  align $4, $2, $3, -1     # CHECK: :[[@LINE]]:21: error: expected 2-bit unsigned immediate
  align $4, $2, $3, 4      # CHECK: :[[@LINE]]:21: error: expected 2-bit unsigned immediate
  beqzc16 $9, 20           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  beqzc16 $6, 31           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  beqzc16 $6, 130          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bnezc16 $9, 20           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  bnezc16 $6, 31           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bnezc16 $6, 130          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  break -1                 # CHECK: :[[@LINE]]:9: error: expected 10-bit unsigned immediate
  break 1024               # CHECK: :[[@LINE]]:9: error: expected 10-bit unsigned immediate
  break -1, 5              # CHECK: :[[@LINE]]:9: error: expected 10-bit unsigned immediate
  break 1024, 5            # CHECK: :[[@LINE]]:9: error: expected 10-bit unsigned immediate
  break 7, -1              # CHECK: :[[@LINE]]:12: error: expected 10-bit unsigned immediate
  break 7, 1024            # CHECK: :[[@LINE]]:12: error: expected 10-bit unsigned immediate
  break 1023, 1024         # CHECK: :[[@LINE]]:15: error: expected 10-bit unsigned immediate
  cache -1, 255($7)        # CHECK: :[[@LINE]]:9: error: expected 5-bit unsigned immediate
  cache 32, 255($7)        # CHECK: :[[@LINE]]:9: error: expected 5-bit unsigned immediate
  # FIXME: Check '0 < pos + size <= 32' constraint on ext
  ext $2, $3, -1, 31       # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ext $2, $3, 32, 31       # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ext $2, $3, 1, 0         # CHECK: :[[@LINE]]:18: error: expected immediate in range 1 .. 32
  ext $2, $3, 1, 33        # CHECK: :[[@LINE]]:18: error: expected immediate in range 1 .. 32
  ins $2, $3, -1, 31       # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ins $2, $3, 32, 31       # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ei $32                   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  swe $33, 8($4)           # CHECK: :[[@LINE]]:7: error: invalid register number
  swe $5, 8($34)           # CHECK: :[[@LINE]]:13: error: invalid register number
  swe $5, 512($4)          # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lapc $7, 1048576         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 19-bit signed immediate and multiple of 4
  lapc $6, -1048580        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 19-bit signed immediate and multiple of 4
  lapc $3, 3               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 19-bit signed immediate and multiple of 4
  lapc $3, -1              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 19-bit signed immediate and multiple of 4
  lbu16 $9, 8($16)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lbu16 $3, -2($16)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lbu16 $3, -2($16)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lbu16 $16, 8($9)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lhu16 $9, 4($16)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lhu16 $3, 64($16)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lhu16 $3, 64($16)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lhu16 $16, 4($9)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  li16 $4, -2              # CHECK: :[[@LINE]]:12: error: expected immediate in range -1 .. 126
  li16 $4, 127             # CHECK: :[[@LINE]]:12: error: expected immediate in range -1 .. 126
  lsa   $4, $2, $3, 0      # CHECK: :[[@LINE]]:21: error: expected immediate in range 1 .. 4
  lsa   $4, $2, $3, 5      # CHECK: :[[@LINE]]:21: error: expected immediate in range 1 .. 4
  lw16  $9, 8($17)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lw16  $4, 68($17)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lw16  $4, 68($17)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lw16  $17, 8($10)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  pref -1, 255($7)         # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  pref 32, 255($7)         # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  teq $34, $9, 5           # CHECK: :[[@LINE]]:7: error: invalid register number
  teq $8, $35, 6           # CHECK: :[[@LINE]]:11: error: invalid register number
  tge $34, $9, 5           # CHECK: :[[@LINE]]:7: error: invalid register number
  tge $8, $35, 6           # CHECK: :[[@LINE]]:11: error: invalid register number
  tgeu $34, $9, 5          # CHECK: :[[@LINE]]:8: error: invalid register number
  tgeu $8, $35, 6          # CHECK: :[[@LINE]]:12: error: invalid register number
  tlt $34, $9, 5           # CHECK: :[[@LINE]]:7: error: invalid register number
  tlt $8, $35, 6           # CHECK: :[[@LINE]]:11: error: invalid register number
  tltu $34, $9, 5          # CHECK: :[[@LINE]]:8: error: invalid register number
  tltu $8, $35, 6          # CHECK: :[[@LINE]]:12: error: invalid register number
  tne $34, $9, 5           # CHECK: :[[@LINE]]:7: error: invalid register number
  tne $8, $35, 6           # CHECK: :[[@LINE]]:11: error: invalid register number
  wait -1                  # CHECK: :[[@LINE]]:8: error: expected 10-bit unsigned immediate
  wait 1024                # CHECK: :[[@LINE]]:8: error: expected 10-bit unsigned immediate
  wrpgpr $34, $4           # CHECK: :[[@LINE]]:10: error: invalid register number
  wrpgpr $3, $33           # CHECK: :[[@LINE]]:14: error: invalid register number
  wsbh $34, $4             # CHECK: :[[@LINE]]:8: error: invalid register number
  wsbh $3, $33             # CHECK: :[[@LINE]]:12: error: invalid register number
  jrcaddiusp 1             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 2             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 3             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 10            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 18            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 31            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 33            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 125           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 132           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  lwm16 $5, $6, $ra, 8($sp)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: $16 or $31 expected
  lwm16 $16, $19, $ra, 8($sp) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: consecutive register numbers expected
  lwm16 $16-$25, $ra, 8($sp)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  lwm16 $16, 8($sp)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lwm16 $16, $17, 8($sp)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lwm16 $16-$20, 8($sp)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lwm16 $16, $17, $ra, 8($fp)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lwm16 $16, $17, $ra, 64($sp) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sb16 $9, 4($16)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sb16 $3, 64($16)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  sb16 $16, 4($16)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sb16 $7, 4($9)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sh16  $9, 8($17)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sh16  $4, 68($17)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  sh16  $16, 8($17)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sh16  $7, 8($9)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sync -1                  # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  sync 32                  # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  sw16  $9, 4($17)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sw16  $4, 64($17)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  sw16  $16, 4($17)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sw16  $7, 4($10)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  swm16 $5, $6, $ra, 8($sp)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: $16 or $31 expected
  swm16 $16, $19, $ra, 8($sp) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: consecutive register numbers expected
  swm16 $16-$25, $ra, 8($sp)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  swm16 $16, 8($sp)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  swm16 $16, $17, 8($sp)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  swm16 $16-$20, 8($sp)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  swm16 $16, $17, $ra, 8($fp)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  swm16 $16, $17, $ra, 64($sp) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  mtc0  $4, $3, -1         # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mtc0  $4, $3, 8          # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mthc0 $4, $3, -1         # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mthc0 $4, $3, 8          # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mfc0  $4, $3, -1         # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mfc0  $4, $3, 8          # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mfhc0 $4, $3, -1         # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mfhc0 $4, $3, 8          # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  tlbp $3                  # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbp 5                   # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbp $4, 6               # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbr $3                  # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbr 5                   # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbr $4, 6               # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbwi $3                 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbwi 5                  # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbwi $4, 6              # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbwr $3                 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbwr 5                  # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbwr $4, 6              # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  dvp 3                    # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  dvp $4, 5                # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  evp 3                    # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  evp $4, 5                # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  jalrc.hb $31             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
  jalrc.hb $31, $31        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
  sll $4, $3, -1           # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sll $4, $3, 32           # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sra $4, $3, -1           # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sra $4, $3, 32           # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  srl $4, $3, -1           # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  srl $4, $3, 32           # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sll $3, -1               # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sll $3, 32               # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sra $3, -1               # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sra $3, 32               # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  srl $3, -1               # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  srl $3, 32               # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  lle $33, 8($5)           # CHECK: :[[@LINE]]:7: error: invalid register number
  lle $4, 8($33)           # CHECK: :[[@LINE]]:13: error: invalid register number
  lle $4, 512($5)          # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lle $4, -513($5)         # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lwe $33, 8($5)           # CHECK: :[[@LINE]]:7: error: invalid register number
  lwe $4, 8($33)           # CHECK: :[[@LINE]]:13: error: invalid register number
  lwe $4, 512($5)          # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lwe $4, -513($5)         # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sbe $33, 8($5)           # CHECK: :[[@LINE]]:7: error: invalid register number
  sbe $4, 8($33)           # CHECK: :[[@LINE]]:13: error: invalid register number
  sbe $4, 512($5)          # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sbe $4, -513($5)         # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sce $33, 8($5)           # CHECK: :[[@LINE]]:7: error: invalid register number
  sce $4, 8($33)           # CHECK: :[[@LINE]]:13: error: invalid register number
  sce $4, 512($5)          # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sce $4, -513($5)         # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  she $33, 8($5)           # CHECK: :[[@LINE]]:7: error: invalid register number
  she $4, 8($33)           # CHECK: :[[@LINE]]:13: error: invalid register number
  she $4, 512($5)          # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  she $4, -513($5)         # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  swe $5, -513($4)         # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lh $33, 8($4)            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lhe $34, 8($2)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lhu $35, 8($2)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lhue $36, 8($2)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lh $2, 8($34)            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lhe $4, 8($33)           # CHECK: :[[@LINE]]:13: error: invalid register number
  lhu $4, 8($35)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lhue $4, 8($37)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register number
  lh $2, -2147483649($4)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 32-bit signed offset
  lh $2, 2147483648($4)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 32-bit signed offset
  lhe $4, -512($2)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lhe $4, 512($2)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lhu $4, -2147483649($2)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 32-bit signed offset
  lhu $4, 2147483648($2)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 32-bit signed offset
  lhue $4, -512($2)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lhue $4, 512($2)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lwm32 $5, $6, 8($4)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: $16 or $31 expected
  lwm32 $16, $19, 8($4)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: consecutive register numbers expected
  lwm32 $16-$25, 8($4)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  lwm32 $16, $17, $18, $19, $20, $21, $22, $23, $24, 8($4) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  movep $5, $6, $2, $9     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  movep $5, $6, $5, $3     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  movep $5, $21, $2, $3    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  movep $8, $6, $2, $3     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  rotr $2, -1              # CHECK: :[[@LINE]]:12: error: expected 5-bit unsigned immediate
  rotr $2, 32              # CHECK: :[[@LINE]]:12: error: expected 5-bit unsigned immediate
  rotr $2, $3, -1          # CHECK: :[[@LINE]]:16: error: expected 5-bit unsigned immediate
  rotr $2, $3, 32          # CHECK: :[[@LINE]]:16: error: expected 5-bit unsigned immediate
  rotrv $9, $6, 5          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  swm32 $5, $6, 8($4)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: $16 or $31 expected
  swm32 $16, $19, 8($4)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: consecutive register numbers expected
  swm32 $16-$25, 8($4)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  lwp $31, 8($4)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
                           # FIXME: This ought to point at the $34 but memory is treated as one operand.
  lwp $16, 8($34)          # CHECK: :[[@LINE]]:14: error: invalid register number
  lwp $16, 4096($4)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 12-bit signed offset
  lwp $16, 8($16)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
  swp $31, 8($4)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  swp $16, 8($34)          # CHECK: :[[@LINE]]:14: error: invalid register number
  swp $16, 4096($4)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 12-bit signed offset
                           # bposge32 is microMIPS DSP instruction
  bposge32 342             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  bc1eqzc $f32, 4          # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  bc1eqzc $f31, -65535     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc1eqzc $f31, -65537     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc1eqzc $f31, 65535      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc1eqzc $f31, 65536      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc1nezc $f32, 4          # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  bc1nezc $f31, -65535     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc1nezc $f31, -65537     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc1nezc $f31, 65535      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc1nezc $f31, 65536      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc2eqzc $32, 4           # CHECK: :[[@LINE]]:11: error: invalid register number
  bc2eqzc $31, -65535      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc2eqzc $31, -65537      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc2eqzc $31, 65535       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc2eqzc $31, 65536       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc2nezc $32, 4           # CHECK: :[[@LINE]]:11: error: invalid register number
  bc2nezc $31, -65535      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc2nezc $31, -65537      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc2nezc $31, 65535       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc2nezc $31, 65536       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  jalrc $31                # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
  jalrc $31, $31           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
  andi $3, $4, -1          # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  andi $3, $4, 65536       # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  andi $3, -1              # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  andi $3, 65536           # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  ori $3, $4, -1           # CHECK: :[[@LINE]]:15: error: expected 16-bit unsigned immediate
  ori $3, $4, 65536        # CHECK: :[[@LINE]]:15: error: expected 16-bit unsigned immediate
  ori $3, -1               # CHECK: :[[@LINE]]:11: error: expected 16-bit unsigned immediate
  ori $3, 65536            # CHECK: :[[@LINE]]:11: error: expected 16-bit unsigned immediate
  xori $3, $4, -1          # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  xori $3, $4, 65536       # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  xori $3, -1              # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  xori $3, 65536           # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  not $3, 4                # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  lb $32, 8($5)            # CHECK: :[[@LINE]]:6: error: invalid register number
  lb $4, -2147483649($5)   # CHECK: :[[@LINE]]:10: error: expected memory with 16-bit signed offset
  lb $4, 2147483648($5)    # CHECK: :[[@LINE]]:10: error: expected memory with 16-bit signed offset
  lb $4, 8($32)            # CHECK: :[[@LINE]]:12: error: invalid register number
  lbu $32, 8($5)           # CHECK: :[[@LINE]]:7: error: invalid register number
  lbu $4, -2147483649($5)  # CHECK: :[[@LINE]]:11: error: expected memory with 16-bit signed offset
  lbu $4, 2147483648($5)   # CHECK: :[[@LINE]]:11: error: expected memory with 16-bit signed offset
  lbu $4, 8($32)           # CHECK: :[[@LINE]]:13: error: invalid register number
  ldc1 $f32, 300($10)      # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  ldc1 $f7, -32769($10)    # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  ldc1 $f7, 32768($10)     # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  ldc1 $f7, 300($32)       # CHECK: :[[@LINE]]:17: error: invalid register number
  sdc1 $f32, 64($10)       # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  sdc1 $f7, -32769($10)    # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  sdc1 $f7, 32768($10)     # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  sdc1 $f7, 64($32)        # CHECK: :[[@LINE]]:16: error: invalid register number
  lwc1 $f32, 32($5)        # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  lwc1 $f2, -32769($5)     # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  lwc1 $f2, 32768($5)      # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  lwc1 $f2, 32($32)        # CHECK: :[[@LINE]]:16: error: invalid register number
  swc1 $f32, 369($13)      # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  swc1 $f6, -32769($13)    # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  swc1 $f6, 32768($13)     # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  swc1 $f6, 369($32)       # CHECK: :[[@LINE]]:17: error: invalid register number
  ldc2 $32, 1023($12)      # CHECK: :[[@LINE]]:8: error: invalid register number
  sdc2 $32, 8($16)         # CHECK: :[[@LINE]]:8: error: invalid register number
  lwc2 $32, 16($4)         # CHECK: :[[@LINE]]:8: error: invalid register number
  swc2 $32, 777($17)       # CHECK: :[[@LINE]]:8: error: invalid register number
  sdc2 $11, -1025($12)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  sdc2 $11, 1024($12)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  swc2 $11, -1025($12)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  swc2 $11, 1024($12)      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  bgec  $0, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  bgec  $2, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
  bgec  $2, $4, -131076    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bgec  $2, $4, -131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bgec  $2, $4, 131072     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bgec  $2, $4, 131071     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bltc  $0, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  bltc  $2, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
  bltc  $2, $4, -131076    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bltc  $2, $4, -131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bltc  $2, $4, 131072     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bltc  $2, $4, 131071     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bgeuc $0, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  bgeuc $2, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
  bgeuc  $2, $4, -131076   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bgeuc  $2, $4, -131071   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bgeuc  $2, $4, 131072    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bgeuc  $2, $4, 131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bltuc $0, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  bltuc $2, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
  bltuc  $2, $4, -131076   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bltuc  $2, $4, -131071   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bltuc  $2, $4, 131072    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bltuc  $2, $4, 131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  beqc  $0, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  beqc  $2, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
  beqc  $2, $4, -131076    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  beqc  $2, $4, -131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  beqc  $2, $4, 131072     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  beqc  $2, $4, 131071     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bnec  $0, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  bnec  $2, $2, 12         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: registers must be different
  bnec  $2, $4, -131076    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bnec  $2, $4, -131071    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bnec  $2, $4, 131072     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bnec  $2, $4, 131071     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  blezc $0, 12             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  blezc $2, -131076        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  blezc $2, -131071        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  blezc $2, 131072         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  blezc $2, 131071         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bgezc $0, 12             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  bgezc $2, -131076        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bgezc $2, -131071        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bgezc $2, 131072         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bgezc $2, 131071         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bgtzc $0, 12             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  bgtzc $2, -131076        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bgtzc $2, -131071        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bgtzc $2, 131072         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bgtzc $2, 131071         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bltzc $0, 12             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  bltzc $2, -131076        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bltzc $2, -131071        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bltzc $2, 131072         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bltzc $2, 131071         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  beqzc $0, 12             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  beqzc $2, -4194308       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  beqzc $2, -4194303       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  beqzc $2, 4194304        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  beqzc $2, 4194303        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bnezc $0, 12             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand ($zero) for instruction
  bnezc $2, -4194308       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bnezc $2, -4194303       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bnezc $2, 4194304        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bnezc $2, 4194303        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  teq $8, $9, $2           # CHECK: :[[@LINE]]:15: error: expected 4-bit unsigned immediate
  teq $8, $9, -1           # CHECK: :[[@LINE]]:15: error: expected 4-bit unsigned immediate
  tge $8, $9, $2           # CHECK: :[[@LINE]]:15: error: expected 4-bit unsigned immediate
  tge $8, $9, -1           # CHECK: :[[@LINE]]:15: error: expected 4-bit unsigned immediate
  tgeu $8, $9, $2          # CHECK: :[[@LINE]]:16: error: expected 4-bit unsigned immediate
  tgeu $8, $9, -1          # CHECK: :[[@LINE]]:16: error: expected 4-bit unsigned immediate
  tlt $8, $9, $2           # CHECK: :[[@LINE]]:15: error: expected 4-bit unsigned immediate
  tlt $8, $9, -1           # CHECK: :[[@LINE]]:15: error: expected 4-bit unsigned immediate
  tltu $8, $9, $2          # CHECK: :[[@LINE]]:16: error: expected 4-bit unsigned immediate
  tltu $8, $9, -1          # CHECK: :[[@LINE]]:16: error: expected 4-bit unsigned immediate
  tne $8, $9, $2           # CHECK: :[[@LINE]]:15: error: expected 4-bit unsigned immediate
  tne $8, $9, -1           # CHECK: :[[@LINE]]:15: error: expected 4-bit unsigned immediate
  teqi $4, 5               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  tgei $4, 5               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  tgeiu $4, 5              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  tlti $4, 5               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  tltiu $4, 5              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  tnei $4, 5               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  syscall -1               # CHECK: :[[@LINE]]:11: error: expected 10-bit unsigned immediate
  syscall $4               # CHECK: :[[@LINE]]:11: error: expected 10-bit unsigned immediate
  ldc2 $1, 1023($32)       # CHECK: :[[@LINE]]:17: error: invalid register number
  lwc2 $1, 16($32)         # CHECK: :[[@LINE]]:15: error: invalid register number
  sdc2 $1, 8($32)          # CHECK: :[[@LINE]]:14: error: invalid register number
  swc2 $1, 777($32)        # CHECK: :[[@LINE]]:16: error: invalid register number
