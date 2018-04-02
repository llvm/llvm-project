# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -iterations=1 -resource-pressure=0 -timeline < %s | FileCheck %s

  imull  %esi
  imull  (%rdi)


# The second integer multiply can start at cycle 2 because the implicit reads
# can start after the load operand is evaluated.

# CHECK: Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]	Instructions:
# CHECK-NEXT:  2      3     1.00                    	imull	%esi
# CHECK-NEXT:  2      6     1.00    *               	imull	(%rdi)


# CHECK:      Timeline view:
# CHECK:      Index	0123456789
# CHECK:      [0,0]	DeeeER   .	imull	%esi
# CHECK-NEXT: [0,1]	.DeeeeeeER	imull	(%rdi)
