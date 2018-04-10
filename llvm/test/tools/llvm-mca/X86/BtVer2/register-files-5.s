# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=1 -resource-pressure=false -instruction-info=false -dispatch-stats -register-file-stats -timeline < %s | FileCheck %s

  vdivps %ymm0, %ymm0, %ymm1
  vaddps %ymm0, %ymm0, %ymm2
  vaddps %ymm0, %ymm0, %ymm3
  vaddps %ymm0, %ymm0, %ymm4
  vaddps %ymm0, %ymm0, %ymm5
  vaddps %ymm0, %ymm0, %ymm6
  vaddps %ymm0, %ymm0, %ymm7
  vaddps %ymm0, %ymm0, %ymm8
  vaddps %ymm0, %ymm0, %ymm9
  vaddps %ymm0, %ymm0, %ymm10
  vaddps %ymm0, %ymm0, %ymm11
  vaddps %ymm0, %ymm0, %ymm12
  vaddps %ymm0, %ymm0, %ymm13
  vaddps %ymm0, %ymm0, %ymm14
  vaddps %ymm0, %ymm0, %ymm15
  vaddps %ymm2, %ymm0, %ymm0
  vaddps %ymm2, %ymm0, %ymm3
  vaddps %ymm2, %ymm0, %ymm4
  vaddps %ymm2, %ymm0, %ymm5
  vaddps %ymm2, %ymm0, %ymm6
  vaddps %ymm2, %ymm0, %ymm7
  vaddps %ymm2, %ymm0, %ymm8
  vaddps %ymm2, %ymm0, %ymm9
  vaddps %ymm2, %ymm0, %ymm10
  vaddps %ymm2, %ymm0, %ymm11
  vaddps %ymm2, %ymm0, %ymm12
  vaddps %ymm2, %ymm0, %ymm13
  vaddps %ymm2, %ymm0, %ymm14
  vaddps %ymm2, %ymm0, %ymm15
  vaddps %ymm3, %ymm0, %ymm2
  vaddps %ymm3, %ymm0, %ymm4
  vaddps %ymm3, %ymm0, %ymm5
  vaddps %ymm3, %ymm0, %ymm6


# CHECK:      Iterations:     1
# CHECK-NEXT: Instructions:   33
# CHECK-NEXT: Total Cycles:   70
# CHECK-NEXT: Dispatch Width: 2
# CHECK-NEXT: IPC:            0.47


# CHECK:      Dynamic Dispatch Stall Cycles:
# CHECK-NEXT: RAT     - Register unavailable:                      0
# CHECK-NEXT: RCU     - Retire tokens unavailable:                 8
# CHECK-NEXT: SCHEDQ  - Scheduler full:                            0
# CHECK-NEXT: LQ      - Load queue full:                           0
# CHECK-NEXT: SQ      - Store queue full:                          0
# CHECK-NEXT: GROUP   - Static restrictions on the dispatch group: 0


# CHECK:      Register File statistics:
# CHECK-NEXT: Total number of mappings created:   66
# CHECK-NEXT: Max number of mappings used:        64

# CHECK:      *  Register File #1 -- FpuPRF:
# CHECK-NEXT:    Number of physical registers:     72
# CHECK-NEXT:    Total number of mappings created: 66
# CHECK-NEXT:    Max number of mappings used:      64

# CHECK:      *  Register File #2 -- IntegerPRF:
# CHECK-NEXT:    Number of physical registers:     64
# CHECK-NEXT:    Total number of mappings created: 0
# CHECK-NEXT:    Max number of mappings used:      0


# CHECK:      Timeline view:
# CHECK-NEXT:      	          0123456789          0123456789          0123456789          
# CHECK-NEXT: Index	0123456789          0123456789          0123456789          0123456789

# CHECK:      [0,0]	DeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeER    .    .    .    .    .   .	vdivps	%ymm0, %ymm0, %ymm1
# CHECK-NEXT: [0,1]	.DeeeE----------------------------------R    .    .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm2
# CHECK-NEXT: [0,2]	. D=eeeE---------------------------------R   .    .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm3
# CHECK-NEXT: [0,3]	.  D==eeeE-------------------------------R   .    .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm4
# CHECK-NEXT: [0,4]	.   D===eeeE------------------------------R  .    .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm5
# CHECK-NEXT: [0,5]	.    D====eeeE----------------------------R  .    .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm6
# CHECK-NEXT: [0,6]	.    .D=====eeeE---------------------------R .    .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm7
# CHECK-NEXT: [0,7]	.    . D======eeeE-------------------------R .    .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm8
# CHECK-NEXT: [0,8]	.    .  D=======eeeE------------------------R.    .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm9
# CHECK-NEXT: [0,9]	.    .   D========eeeE----------------------R.    .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm10
# CHECK-NEXT: [0,10]	.    .    D=========eeeE---------------------R    .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm11
# CHECK-NEXT: [0,11]	.    .    .D==========eeeE-------------------R    .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm12
# CHECK-NEXT: [0,12]	.    .    . D===========eeeE------------------R   .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm13
# CHECK-NEXT: [0,13]	.    .    .  D============eeeE----------------R   .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm14
# CHECK-NEXT: [0,14]	.    .    .   D=============eeeE---------------R  .    .    .    .   .	vaddps	%ymm0, %ymm0, %ymm15
# CHECK-NEXT: [0,15]	.    .    .    D==============eeeE-------------R  .    .    .    .   .	vaddps	%ymm2, %ymm0, %ymm0
# CHECK-NEXT: [0,16]	.    .    .    .D================eeeE-----------R .    .    .    .   .	vaddps	%ymm2, %ymm0, %ymm3
# CHECK-NEXT: [0,17]	.    .    .    . D=================eeeE---------R .    .    .    .   .	vaddps	%ymm2, %ymm0, %ymm4
# CHECK-NEXT: [0,18]	.    .    .    .  D==================eeeE--------R.    .    .    .   .	vaddps	%ymm2, %ymm0, %ymm5
# CHECK-NEXT: [0,19]	.    .    .    .   D===================eeeE------R.    .    .    .   .	vaddps	%ymm2, %ymm0, %ymm6
# CHECK-NEXT: [0,20]	.    .    .    .    D====================eeeE-----R    .    .    .   .	vaddps	%ymm2, %ymm0, %ymm7
# CHECK-NEXT: [0,21]	.    .    .    .    .D=====================eeeE---R    .    .    .   .	vaddps	%ymm2, %ymm0, %ymm8
# CHECK-NEXT: [0,22]	.    .    .    .    . D======================eeeE--R   .    .    .   .	vaddps	%ymm2, %ymm0, %ymm9
# CHECK-NEXT: [0,23]	.    .    .    .    .  D=======================eeeER   .    .    .   .	vaddps	%ymm2, %ymm0, %ymm10
# CHECK-NEXT: [0,24]	.    .    .    .    .   D========================eeeER .    .    .   .	vaddps	%ymm2, %ymm0, %ymm11
# CHECK-NEXT: [0,25]	.    .    .    .    .    D=========================eeeER    .    .   .	vaddps	%ymm2, %ymm0, %ymm12
# CHECK-NEXT: [0,26]	.    .    .    .    .    .D==========================eeeER  .    .   .	vaddps	%ymm2, %ymm0, %ymm13
# CHECK-NEXT: [0,27]	.    .    .    .    .    . D===========================eeeER.    .   .	vaddps	%ymm2, %ymm0, %ymm14
# CHECK-NEXT: [0,28]	.    .    .    .    .    .  D============================eeeER   .   .	vaddps	%ymm2, %ymm0, %ymm15
# CHECK-NEXT: [0,29]	.    .    .    .    .    .   D=============================eeeER .   .	vaddps	%ymm3, %ymm0, %ymm2
# CHECK-NEXT: [0,30]	.    .    .    .    .    .    D==============================eeeER   .	vaddps	%ymm3, %ymm0, %ymm4
# CHECK-NEXT: [0,31]	.    .    .    .    .    .    .D===============================eeeER .	vaddps	%ymm3, %ymm0, %ymm5
# CHECK-NEXT: [0,32]	.    .    .    .    .    .    .    .    D========================eeeER	vaddps	%ymm3, %ymm0, %ymm6
