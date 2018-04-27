# Instructions that are invalid.
#
# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r5 -mattr=+virt 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -arch=mips64 -mcpu=mips64r5 -mattr=+virt 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r5 -mattr=+micromips,+virt 2>%t1
# RUN: FileCheck %s < %t1

  mfgc0                   # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  mfgc0 0                 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  mfgc0 $4                # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  mfgc0 0, $4             # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  mfgc0 0, $4, $5         # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  mfgc0 $4, 0, $5         # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
  mfgc0 $4, $5, 8         # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mfgc0 $4, $5, -1        # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mfgc0 $4, $5, 0($4)     # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
  mtgc0                   # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  mtgc0 0                 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  mtgc0 $4                # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  mtgc0 0, $4             # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  mtgc0 0, $4, $5         # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  mtgc0 $4, 0, $5         # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
  mtgc0 $4, $5, 8         # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mtgc0 $4, $5, -1        # CHECK: :[[@LINE]]:17: error: expected 3-bit unsigned immediate
  mtgc0 $4, $5, 0($4)     # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
  mfhgc0                  # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  mfhgc0 0                # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  mfhgc0 $4               # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  mfhgc0 0, $4            # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  mfhgc0 0, $4, $5        # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  mfhgc0 $4, 0, $5        # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  mfhgc0 $4, $5, 8        # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  mfhgc0 $4, $5, -1       # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  mfhgc0 $4, $5, 0($4)    # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
  mthgc0                  # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  mthgc0 0                # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  mthgc0 $4               # CHECK: :[[@LINE]]:3: error: too few operands for instruction
  mthgc0 0, $4            # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  mthgc0 0, $4, $5        # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  mthgc0 $4, 0, $5        # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  mthgc0 $4, $5, 8        # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  mthgc0 $4, $5, -1       # CHECK: :[[@LINE]]:18: error: expected 3-bit unsigned immediate
  mthgc0 $4, $5, 0($4)    # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
  hypcall $4              # CHECK: :[[@LINE]]:11: error: expected 10-bit unsigned immediate
  hypcall 0, $4           # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  hypcall 0, $4, $5       # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
  hypcall $4, 0, $5       # CHECK: :[[@LINE]]:11: error: expected 10-bit unsigned immediate
  hypcall $4, $5, 8       # CHECK: :[[@LINE]]:11: error: expected 10-bit unsigned immediate
  hypcall $4, $5, -1      # CHECK: :[[@LINE]]:11: error: expected 10-bit unsigned immediate
  hypcall $4, $5, 0($4)   # CHECK: :[[@LINE]]:11: error: expected 10-bit unsigned immediate
  hypcall 2048            # CHECK: :[[@LINE]]:11: error: expected 10-bit unsigned immediate
  hypcall -1              # CHECK: :[[@LINE]]:11: error: expected 10-bit unsigned immediate
  hypcall 0($4)           # CHECK: :[[@LINE]]:12: error: unexpected token in argument list
  tlbginv 0               # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  tlbginv $4              # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  tlbginv 0, $4           # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  tlbginv 0, $4, $5       # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  tlbginv $4, 0, $5       # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  tlbginv $4, $5, 8       # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  tlbginv $4, $5, -1      # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  tlbginv $4, $5, 0($4)   # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
  tlbginvf 0              # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  tlbginvf $4             # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  tlbginvf 0, $4          # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  tlbginvf 0, $4, $5      # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  tlbginvf $4, 0, $5      # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  tlbginvf $4, $5, 8      # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  tlbginvf $4, $5, -1     # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  tlbginvf $4, $5, 0($4)  # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
  tlbgp 0                 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgp $4                # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgp 0, $4             # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgp 0, $4, $5         # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgp $4, 0, $5         # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgp $4, $5, 8         # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgp $4, $5, -1        # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgp $4, $5, 0($4)     # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgr 0                 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgr $4                # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgr 0, $4             # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgr 0, $4, $5         # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgr $4, 0, $5         # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgr $4, $5, 8         # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgr $4, $5, -1        # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgr $4, $5, 0($4)     # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
  tlbgwi 0                # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwi $4               # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwi 0, $4            # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwi 0, $4, $5        # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwi $4, 0, $5        # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwi $4, $5, 8        # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwi $4, $5, -1       # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwi $4, $5, 0($4)    # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwr 0                # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwr $4               # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwr 0, $4            # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwr 0, $4, $5        # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwr $4, 0, $5        # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwr $4, $5, 8        # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwr $4, $5, -1       # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
  tlbgwr $4, $5, 0($4)    # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
