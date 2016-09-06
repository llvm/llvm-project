# RUN: not llvm-mc %s -triple=mips -show-encoding -mcpu=mips64r6 -mattr=micromips 2>%t1
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
  cache -1, 255($7)        # CHECK: :[[@LINE]]:9: error: expected 5-bit unsigned immediate
  cache 32, 255($7)        # CHECK: :[[@LINE]]:9: error: expected 5-bit unsigned immediate
  # FIXME: Check various 'pos + size' constraints on dext*
  dext $2, $3, -1, 1   # CHECK: :[[@LINE]]:16: error: expected 6-bit unsigned immediate
  dext $2, $3, 64, 1   # CHECK: :[[@LINE]]:16: error: expected 6-bit unsigned immediate
  dext $2, $3, 1, 0    # CHECK: :[[@LINE]]:19: error: expected immediate in range 1 .. 32
  dext $2, $3, 1, 33   # CHECK: :[[@LINE]]:19: error: expected immediate in range 1 .. 32
  dextm $2, $3, -1, 1  # CHECK: :[[@LINE]]:17: error: expected 5-bit unsigned immediate
  dextm $2, $3, 32, 1  # CHECK: :[[@LINE]]:17: error: expected 5-bit unsigned immediate
  dextm $2, $3, -1, 33 # CHECK: :[[@LINE]]:17: error: expected 5-bit unsigned immediate
  dextm $2, $3, 32, 33 # CHECK: :[[@LINE]]:17: error: expected 5-bit unsigned immediate
  dextm $2, $3, 1, 32  # CHECK: :[[@LINE]]:20: error: expected immediate in range 33 .. 64
  dextm $2, $3, 1, 65  # CHECK: :[[@LINE]]:20: error: expected immediate in range 33 .. 64
  dextu $2, $3, 31, 1  # CHECK: :[[@LINE]]:17: error: expected immediate in range 32 .. 63
  dextu $2, $3, 64, 1  # CHECK: :[[@LINE]]:17: error: expected immediate in range 32 .. 63
  dextu $2, $3, 32, 0  # CHECK: :[[@LINE]]:21: error: expected immediate in range 1 .. 32
  dextu $2, $3, 32, 33 # CHECK: :[[@LINE]]:21: error: expected immediate in range 1 .. 32
  dins $2, $3, 31, 33  # CHECK: :[[@LINE]]:20: error: expected immediate in range 1 .. 32
  dins $2, $3, 31, 0   # CHECK: :[[@LINE]]:20: error: expected immediate in range 1 .. 32
  # FIXME: Check '32 <= pos + size <= 64' constraint on dinsm
  dinsm $2, $3, -1, 1  # CHECK: :[[@LINE]]:17: error: expected 5-bit unsigned immediate
  dinsm $2, $3, 32, 1  # CHECK: :[[@LINE]]:17: error: expected 5-bit unsigned immediate
  dinsm $2, $3, 31, 0  # CHECK: :[[@LINE]]:21: error: expected immediate in range 2 .. 64
  dinsm $2, $3, 31, 65 # CHECK: :[[@LINE]]:21: error: expected immediate in range 2 .. 64
  dinsu $2, $3, 31, 1  # CHECK: :[[@LINE]]:17: error: expected immediate in range 32 .. 63
  dinsu $2, $3, 64, 1  # CHECK: :[[@LINE]]:17: error: expected immediate in range 32 .. 63
  dinsu $2, $3, 63, 0  # CHECK: :[[@LINE]]:21: error: expected immediate in range 1 .. 32
  dinsu $2, $3, 32, 33 # CHECK: :[[@LINE]]:21: error: expected immediate in range 1 .. 32
  # FIXME: Check '0 < pos + size <= 32' constraint on ext
  ext $2, $3, -1, 31       # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ext $2, $3, 32, 31       # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ext $2, $3, 1, 0         # CHECK: :[[@LINE]]:18: error: expected immediate in range 1 .. 32
  ext $2, $3, 1, 33        # CHECK: :[[@LINE]]:18: error: expected immediate in range 1 .. 32
  ins $2, $3, -1, 31       # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  ins $2, $3, 32, 31       # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  dalign  $4, $2, $3, -1   # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
  dalign  $4, $2, $3, 8    # CHECK: :[[@LINE]]:23: error: expected 3-bit unsigned immediate
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
  ddiv $32, $4, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddiv $3, $34, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddiv $3, $4, $35         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmod $32, $4, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmod $3, $34, $5         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmod $3, $4, $35         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddivu $32, $4, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddivu $3, $34, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  ddivu $3, $4, $35        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmodu $32, $4, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmodu $3, $34, $5        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  dmodu $3, $4, $35        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  pref -1, 255($7)         # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  pref 32, 255($7)         # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  teq $34, $9, 5           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  teq $8, $35, 6           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tge $34, $9, 5           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tge $8, $35, 6           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tgeu $34, $9, 5          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tgeu $8, $35, 6          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tlt $34, $9, 5           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tlt $8, $35, 6           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tltu $34, $9, 5          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tltu $8, $35, 6          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tne $34, $9, 5           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  tne $8, $35, 6           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  wrpgpr $34, $4           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  wrpgpr $3, $33           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  wsbh $34, $4             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  wsbh $3, $33             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  jrcaddiusp 1             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 2             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 3             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 10            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 18            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 31            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 33            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 125           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jrcaddiusp 128           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
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
  mtc0  $4, $3, -1             # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mtc0  $4, $3, 8              # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mthc0 $4, $3, -1             # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mthc0 $4, $3, 8              # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  dmtc0  $4, $3, -1            # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  dmtc0  $4, $3, 8             # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  dmfc0  $4, $3, -1            # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  dmfc0  $4, $3, 8             # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  tlbp $3                      # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbp 5                       # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbp $4, 6                   # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbr $3                      # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbr 5                       # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbr $4, 6                   # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  tlbwi $3                     # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbwi 5                      # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbwi $4, 6                  # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbwr $3                     # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbwr 5                      # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbwr $4, 6                  # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  dvp 3                        # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  dvp $4, 5                    # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  evp 3                        # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  evp $4, 5                    # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  jalrc.hb $31                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
  jalrc.hb $31, $31            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
  sll $4, $3, -1               # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sll $4, $3, 32               # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sra $4, $3, -1               # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sra $4, $3, 32               # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  srl $4, $3, -1               # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  srl $4, $3, 32               # CHECK: :[[@LINE]]:15: error: expected 5-bit unsigned immediate
  sll $3, -1                   # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sll $3, 32                   # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sra $3, -1                   # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  sra $3, 32                   # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  srl $3, -1                   # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  srl $3, 32                   # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
  dneg $7, 5                   # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  dneg 4                       # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  dnegu $1, 3                  # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
  dnegu 7                      # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  lle $33, 8($5)               # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  lle $4, 8($33)               # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lle $4, 512($5)              # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lle $4, -513($5)             # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lwe $33, 8($5)               # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  lwe $4, 8($33)               # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lwe $4, 512($5)              # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lwe $4, -513($5)             # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sbe $33, 8($5)               # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  sbe $4, 8($33)               # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sbe $4, 512($5)              # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sbe $4, -513($5)             # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sce $33, 8($5)               # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  sce $4, 8($33)               # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sce $4, 512($5)              # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  sce $4, -513($5)             # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  she $33, 8($5)               # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  she $4, 8($33)               # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  she $4, 512($5)              # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  she $4, -513($5)             # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  swe $33, 8($4)               # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  swe $5, 8($34)               # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  swe $5, 512($4)              # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  swe $5, -513($4)             # CHECK: :[[@LINE]]:11: error: expected memory with 9-bit signed offset
  lh $33, 8($4)                # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lhe $34, 8($2)               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lhu $35, 8($2)               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lhue $36, 8($2)              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lh $2, 8($34)                # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  lhe $4, 8($33)               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lhu $4, 8($35)               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  lhue $4, 8($37)              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lh $2, -65536($4)            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  lh $2, 65536($4)             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  lhe $4, -512($2)             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lhe $4, 512($2)              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lhu $4, -65536($2)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  lhu $4, 65536($2)            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  lhue $4, -512($2)            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lhue $4, 512($2)             # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 9-bit signed offset
  lwm32 $5, $6, 8($4)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: $16 or $31 expected
  lwm32 $16, $19, 8($4)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: consecutive register numbers expected
  lwm32 $16-$25, 8($4)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  lwm32 $16, $17, $18, $19, $20, $21, $22, $23, $24, 8($4) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  movep $5, $6, $2, $9         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  movep $5, $6, $5, $3         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  movep $5, $21, $2, $3        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  movep $8, $6, $2, $3         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  rotr $2, -1                  # CHECK: :[[@LINE]]:12: error: expected 5-bit unsigned immediate
  rotr $2, 32                  # CHECK: :[[@LINE]]:12: error: expected 5-bit unsigned immediate
  rotr $2, $3, -1              # CHECK: :[[@LINE]]:16: error: expected 5-bit unsigned immediate
  rotr $2, $3, 32              # CHECK: :[[@LINE]]:16: error: expected 5-bit unsigned immediate
  rotrv $9, $6, 5              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  swm32 $5, $6, 8($4)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: $16 or $31 expected
  swm32 $16, $19, 8($4)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: consecutive register numbers expected
  swm32 $16-$25, 8($4)         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  lwp $31, 8($4)               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
                               # FIXME: This ought to point at the $34 but memory is treated as one operand.
  lwp $16, 8($34)              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 12-bit signed offset
  lwp $16, 4096($4)            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 12-bit signed offset
  lwp $16, 8($16)              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: source and destination must be different
  swp $31, 8($4)               # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  swp $16, 8($34)              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 12-bit signed offset
  swp $16, 4096($4)            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 12-bit signed offset
  dsll $3, $4, 64              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 6-bit unsigned immediate
  dsll $3, $4, -1              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 6-bit unsigned immediate
  dsll32 $3, $4, 32            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 5-bit unsigned immediate
  dsll32 $3, $4, -1            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 5-bit unsigned immediate
  dsra $4, $5, 64              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 6-bit unsigned immediate
  dsra $4, $5, -1              # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 6-bit unsigned immediate
  dsra32 $4, $5, 32            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 5-bit unsigned immediate
  dsra32 $4, $5, -1            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 5-bit unsigned immediate
                               # bposge32 is microMIPS DSP instruction
  bposge32 342                 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
  bc1eqzc $f32, 4              # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  bc1eqzc $f31, -65535         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc1eqzc $f31, -65537         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc1eqzc $f31, 65535          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc1eqzc $f31, 65536          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc1nezc $f32, 4              # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  bc1nezc $f31, -65535         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc1nezc $f31, -65537         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc1nezc $f31, 65535          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc1nezc $f31, 65536          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc2eqzc $32, 4               # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  bc2eqzc $31, -65535          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc2eqzc $31, -65537          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc2eqzc $31, 65535           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc2eqzc $31, 65536           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc2nezc $32, 4               # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  bc2nezc $31, -65535          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc2nezc $31, -65537          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  bc2nezc $31, 65535           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch to misaligned address
  bc2nezc $31, 65536           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: branch target out of range
  andi $3, $4, -1              # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  andi $3, $4, 65536           # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  andi $3, -1                  # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  andi $3, 65536               # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  ori $3, $4, -1               # CHECK: :[[@LINE]]:15: error: expected 16-bit unsigned immediate
  ori $3, $4, 65536            # CHECK: :[[@LINE]]:15: error: expected 16-bit unsigned immediate
  ori $3, -1                   # CHECK: :[[@LINE]]:11: error: expected 16-bit unsigned immediate
  ori $3, 65536                # CHECK: :[[@LINE]]:11: error: expected 16-bit unsigned immediate
  xori $3, $4, -1              # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  xori $3, $4, 65536           # CHECK: :[[@LINE]]:16: error: expected 16-bit unsigned immediate
  xori $3, -1                  # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  xori $3, 65536               # CHECK: :[[@LINE]]:12: error: expected 16-bit unsigned immediate
  not $3, 4                    # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  drotr $5, $10, 64            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 6-bit unsigned immediate
  drotr $5, $10, -1            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 6-bit unsigned immediate
  drotr32 $1, $2, 32           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 5-bit unsigned immediate
  drotr32 $1, $2, -1           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected 5-bit unsigned immediate
  ld $31, 65536($31)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  ld $31, 32768($31)           # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  ld $31, -32769($31)          # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected memory with 16-bit signed offset
  sd $31, 65536($31)           # CHECK: :[[@LINE]]:11: error: expected memory with 16-bit signed offset
  sd $31, 32768($31)           # CHECK: :[[@LINE]]:11: error: expected memory with 16-bit signed offset
  sd $31, -32769($31)          # CHECK: :[[@LINE]]:11: error: expected memory with 16-bit signed offset
  lb $32, 8($5)                # CHECK: :[[@LINE]]:6: error: invalid operand for instruction
  lb $4, -32769($5)            # CHECK: :[[@LINE]]:10: error: expected memory with 16-bit signed offset
  lb $4, 32768($5)             # CHECK: :[[@LINE]]:10: error: expected memory with 16-bit signed offset
  lb $4, 8($32)                # CHECK: :[[@LINE]]:10: error: expected memory with 16-bit signed offset
  lbu $32, 8($5)               # CHECK: :[[@LINE]]:7: error: invalid operand for instruction
  lbu $4, -32769($5)           # CHECK: :[[@LINE]]:11: error: expected memory with 16-bit signed offset
  lbu $4, 32768($5)            # CHECK: :[[@LINE]]:11: error: expected memory with 16-bit signed offset
  lbu $4, 8($32)               # CHECK: :[[@LINE]]:11: error: expected memory with 16-bit signed offset
  ldc1 $f32, 300($10)          # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  ldc1 $f7, -32769($10)        # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  ldc1 $f7, 32768($10)         # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  ldc1 $f7, 300($32)           # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  sdc1 $f32, 64($10)           # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  sdc1 $f7, -32769($10)        # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  sdc1 $f7, 32768($10)         # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  sdc1 $f7, 64($32)            # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  lwc1 $f32, 32($5)            # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  lwc1 $f2, -32769($5)         # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  lwc1 $f2, 32768($5)          # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  lwc1 $f2, 32($32)            # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  swc1 $f32, 369($13)          # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  swc1 $f6, -32769($13)        # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  swc1 $f6, 32768($13)         # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  swc1 $f6, 369($32)           # CHECK: :[[@LINE]]:13: error: expected memory with 16-bit signed offset
  ldc2 $32, 1023($12)          # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  sdc2 $32, 8($16)             # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  lwc2 $32, 16($4)             # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
  swc2 $32, 777($17)           # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
