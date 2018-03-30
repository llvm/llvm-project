# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s

vaddps %ymm0, %ymm0, %ymm1
vandps (%rdi), %ymm1, %ymm2

# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]	Instructions:
# CHECK-NEXT:  2      3     2.00                    	vaddps	%ymm0, %ymm0, %ymm1
# CHECK-NEXT:  2      6     2.00    *               	vandps	(%rdi), %ymm1, %ymm2


# CHECK:      Timeline view:
# CHECK:      Index	0123456789
# CHECK:      [0,0]	DeeeER   .	vaddps	%ymm0, %ymm0, %ymm1
# CHECK-NEXT: [0,1]	.DeeeeeeER	vandps	(%rdi), %ymm1, %ymm2
