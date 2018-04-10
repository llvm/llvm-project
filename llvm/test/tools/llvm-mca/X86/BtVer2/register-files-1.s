# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=5 -verbose -instruction-info=false -register-file-stats -timeline < %s | FileCheck %s

vaddps %xmm0, %xmm0, %xmm0
vmulps %xmm0, %xmm0, %xmm0

# CHECK: Iterations:     5
# CHECK-NEXT: Instructions:   10


# CHECK:      Dynamic Dispatch Stall Cycles:
# CHECK-NEXT: RAT     - Register unavailable:                      0
# CHECK-NEXT: RCU     - Retire tokens unavailable:                 0
# CHECK-NEXT: SCHEDQ  - Scheduler full:                            0
# CHECK-NEXT: LQ      - Load queue full:                           0
# CHECK-NEXT: SQ      - Store queue full:                          0
# CHECK-NEXT: GROUP   - Static restrictions on the dispatch group: 0


# CHECK:      Register File statistics:
# CHECK-NEXT: Total number of mappings created:   10
# CHECK-NEXT: Max number of mappings used:        10

# CHECK:      *  Register File #1 -- FpuPRF:
# CHECK-NEXT:    Number of physical registers:     72
# CHECK-NEXT:    Total number of mappings created: 10
# CHECK-NEXT:    Max number of mappings used:      10

# CHECK:      *  Register File #2 -- IntegerPRF:
# CHECK-NEXT:    Number of physical registers:     64
# CHECK-NEXT:    Total number of mappings created: 0
# CHECK-NEXT:    Max number of mappings used:      0


# CHECK: Timeline view:
# CHECK-NEXT:     	          0123456789        
# CHECK-NEXT: Index	0123456789          01234567

# CHECK:      [0,0]	DeeeER    .    .    .    . .	vaddps	%xmm0, %xmm0, %xmm0
# CHECK-NEXT: [0,1]	D===eeER  .    .    .    . .	vmulps	%xmm0, %xmm0, %xmm0
# CHECK-NEXT: [1,0]	.D====eeeER    .    .    . .	vaddps	%xmm0, %xmm0, %xmm0
# CHECK-NEXT: [1,1]	.D=======eeER  .    .    . .	vmulps	%xmm0, %xmm0, %xmm0
# CHECK-NEXT: [2,0]	. D========eeeER    .    . .	vaddps	%xmm0, %xmm0, %xmm0
# CHECK-NEXT: [2,1]	. D===========eeER  .    . .	vmulps	%xmm0, %xmm0, %xmm0
# CHECK-NEXT: [3,0]	.  D============eeeER    . .	vaddps	%xmm0, %xmm0, %xmm0
# CHECK-NEXT: [3,1]	.  D===============eeER  . .	vmulps	%xmm0, %xmm0, %xmm0
# CHECK-NEXT: [4,0]	.   D================eeeER .	vaddps	%xmm0, %xmm0, %xmm0
# CHECK-NEXT: [4,1]	.   D===================eeER	vmulps	%xmm0, %xmm0, %xmm0
