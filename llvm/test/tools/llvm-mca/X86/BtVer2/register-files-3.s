# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -register-file-size=5 -iterations=2 -verbose -timeline < %s | FileCheck %s

idiv %eax

# CHECK:      Iterations:     2
# CHECK-NEXT: Instructions:   2

# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]	Instructions:
# CHECK-NEXT:  2      25    25.00                 * 	idivl	%eax


# CHECK: RAT     - Register unavailable:                      26

# CHECK:       Register File statistics.
# CHECK-NEXT:  Register File #0
# CHECK-NEXT:    Total number of mappings created: 6
# CHECK-NEXT:    Max number of mappings used:      3


# CHECK:      Timeline view:
# CHECK-NEXT:    	          0123456789          0123456789          01234
# CHECK-NEXT: Index	0123456789          0123456789          0123456789     

# CHECK:      [0,0]	DeeeeeeeeeeeeeeeeeeeeeeeeeER  .    .    .    .    .   .	idivl	%eax
# CHECK:      [1,0]	.    .    .    .    .    . DeeeeeeeeeeeeeeeeeeeeeeeeeER	idivl	%eax
