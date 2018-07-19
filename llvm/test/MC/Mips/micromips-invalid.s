# RUN: not llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips 2>%t1
# RUN: FileCheck %s < %t1

  addiur1sp $7, 260 # CHECK: :[[@LINE]]:17: error: expected both 8-bit unsigned immediate and multiple of 4
  addiur1sp $7, 241 # CHECK: :[[@LINE]]:17: error: expected both 8-bit unsigned immediate and multiple of 4
  addiur1sp $8, 240 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
  addiusp 1032   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addu16  $6, $14, $4 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  subu16  $5, $16, $9 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  andi16  $16, $10, 0x1f # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  andi16  $16, $2, 17 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  and16   $16, $8   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  not16   $18, $9   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  or16    $16, $10  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  xor16   $15, $5   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sll16   $1, $16, 5 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  srl16   $4, $9, 6  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sll16   $3, $16, 9 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  srl16   $4, $5, 15 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addiur2 $9, $7, -1 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  addiur2 $6, $7, 10 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lwm16   $5, $6, $ra, 8($sp)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: $16 or $31 expected
  lwm16   $16, $19, $ra, 8($sp)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: consecutive register numbers expected
  lwm16   $16-$25, $ra, 8($sp)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  lwm16   $16, 8($sp)            # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lwm16   $16, $17, 8($sp)       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lwm16   $16-$20, 8($sp)        # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  swm16   $5, $6, $ra, 8($sp)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: $16 or $31 expected
  swm16   $16, $19, $ra, 8($sp)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: consecutive register numbers expected
  swm16   $16-$25, $ra, 8($sp)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  lwm32   $5, $6, 8($4)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: $16 or $31 expected
  lwm32   $16, $19, 8($4)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: consecutive register numbers expected
  lwm32   $16-$25, 8($4)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  swm32   $5, $6, 8($4)    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: $16 or $31 expected
  swm32   $16, $19, 8($4)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: consecutive register numbers expected
  swm32   $16-$25, 8($4)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  lwm32 $16, $17, $18, $19, $20, $21, $22, $23, $24, 8($4) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid register operand
  addiupc $7, 16777216  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addiupc $6, -16777220 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  addiupc $3, 3         # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lbu16 $9, 8($16) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lhu16 $9, 4($16) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lw16  $9, 8($17) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sb16  $9, 4($16) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sh16  $9, 8($17) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sw16  $9, 4($17) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lbu16 $3, -2($16) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lhu16 $3, 64($16) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lw16  $4, 68($17) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  sb16  $3, 64($16) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  sh16  $4, 68($17) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  sw16  $4, 64($17) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lbu16 $3, -2($16) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lhu16 $3, 64($16) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  lw16  $4, 68($17) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate operand value out of range
  sb16  $16, 4($16) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sh16  $16, 8($17) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sw16  $16, 4($17) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lbu16 $16, 8($9)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lhu16 $16, 4($9)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  lw16  $17, 8($10) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sb16  $7, 4($9)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sh16  $7, 8($9)   # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  sw16  $7, 4($10)  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  cache -1, 8($5)  # CHECK: :[[@LINE]]:9: error: expected 5-bit unsigned immediate
  cache 32, 8($5)  # CHECK: :[[@LINE]]:9: error: expected 5-bit unsigned immediate
  pref -1, 8($5)   # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  pref 32, 8($5)   # CHECK: :[[@LINE]]:8: error: expected 5-bit unsigned immediate
  beqz16 $9, 20 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  bnez16 $9, 20 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  movep   $5, $21, $2, $3 # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  movep   $8, $6, $2, $3  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  movep   $5, $6, $5, $3  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  movep   $5, $6, $2, $9  # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
  break 1024        # CHECK: :[[@LINE]]:9: error: expected 10-bit unsigned immediate
  break 1024, 5     # CHECK: :[[@LINE]]:9: error: expected 10-bit unsigned immediate
  break 7, 1024     # CHECK: :[[@LINE]]:12: error: expected 10-bit unsigned immediate
  break 1024, 1024  # CHECK: :[[@LINE]]:9: error: expected 10-bit unsigned immediate
  wait 1024         # CHECK: :[[@LINE]]:8: error: expected 10-bit unsigned immediate
  prefx -1, $8($5)  # CHECK: :[[@LINE]]:9: error: expected 5-bit unsigned immediate
  prefx 32, $8($5)  # CHECK: :[[@LINE]]:9: error: expected 5-bit unsigned immediate
  jraddiusp 1       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 2       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 3       # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 10      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 18      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 31      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 33      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 125     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  jraddiusp 132     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: expected both 7-bit unsigned immediate and multiple of 4
  lwu $32, 4096($32)     # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
