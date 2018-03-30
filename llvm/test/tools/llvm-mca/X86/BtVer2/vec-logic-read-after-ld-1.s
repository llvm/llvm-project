# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=1 -timeline -resource-pressure=false < %s | FileCheck %s

vaddps %xmm0, %xmm0, %xmm1
vandps (%rdi), %xmm1, %xmm2

# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]	Instructions:
# CHECK-NEXT:  1      3     1.00                    	vaddps	%xmm0, %xmm0, %xmm1
# CHECK-NEXT:  1      6     1.00    *               	vandps	(%rdi), %xmm1, %xmm2


# CHECK:      Timeline view:
# CHECK:      Index	012345678
# CHECK:      [0,0]	DeeeER  .	vaddps	%xmm0, %xmm0, %xmm1
# CHECK-NEXT: [0,1]	DeeeeeeER	vandps	(%rdi), %xmm1, %xmm2
