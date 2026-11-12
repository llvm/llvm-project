# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -skip-unsupported-instructions=any -timeline %s 2>&1 | FileCheck --check-prefix=CHECK-SKIP %s
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -skip-unsupported-instructions=lack-sched -timeline %s 2>&1 | FileCheck --check-prefix=CHECK-SKIP %s
# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -skip-unsupported-instructions=parse-failure -timeline %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

# Test checks that unsupported instructions exit with an error, unless -skip-unsupported-instructions=lack-sched is passed, in which case the remaining instructions should be analysed.
# Additionally check that -skip-unsupported-instructions=parse-failure continues to raise the lack of scheduling information.

# CHECK-SKIP: warning: found an unsupported instruction in the input assembly sequence, skipping with -skip-unsupported-instructions, note accuracy will be impacted:
# CHECK-ERROR: error: found an unsupported instruction in the input assembly sequence, use -skip-unsupported-instructions=lack-sched to ignore these on the input.

bzhi %eax, %ebx, %ecx

# Supported instruction that may be analysed.
add %eax, %eax

# CHECK-SKIP: Iterations:        100
# CHECK-SKIP: Instructions:      100
# CHECK-SKIP: Total Cycles:      103
# CHECK-SKIP: Total uOps:        100

# CHECK-SKIP: Dispatch Width:    2
# CHECK-SKIP: uOps Per Cycle:    0.97
# CHECK-SKIP: IPC:               0.97
# CHECK-SKIP: Block RThroughput: 0.5

# CHECK-SKIP: Instruction Info:
# CHECK-SKIP: [1]: #uOps
# CHECK-SKIP: [2]: Latency
# CHECK-SKIP: [3]: RThroughput
# CHECK-SKIP: [4]: MayLoad
# CHECK-SKIP: [5]: MayStore
# CHECK-SKIP: [6]: HasSideEffects (U)

# CHECK-SKIP: [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK-SKIP:  1      1     0.50                        addl  %eax, %eax

# CHECK-SKIP: Timeline view:

# CHECK-SKIP: [0,0]     DeER .    . .   addl  %eax, %eax
# CHECK-SKIP: [1,0]     D=eER.    . .   addl  %eax, %eax
# CHECK-SKIP: [2,0]     .D=eER    . .   addl  %eax, %eax
# CHECK-SKIP: [3,0]     .D==eER   . .   addl  %eax, %eax
# CHECK-SKIP: [4,0]     . D==eER  . .   addl  %eax, %eax
# CHECK-SKIP: [5,0]     . D===eER . .   addl  %eax, %eax
# CHECK-SKIP: [6,0]     .  D===eER. .   addl  %eax, %eax
# CHECK-SKIP: [7,0]     .  D====eER .   addl  %eax, %eax
# CHECK-SKIP: [8,0]     .   D====eER.   addl  %eax, %eax
# CHECK-SKIP: [9,0]     .   D=====eER   addl  %eax, %eax

# CHECK-SKIP: Average Wait times (based on the timeline view):
# CHECK-SKIP: [0]: Executions
# CHECK-SKIP: [1]: Average time spent waiting in a scheduler's queue
# CHECK-SKIP: [2]: Average time spent waiting in a scheduler's queue while ready
# CHECK-SKIP: [3]: Average time elapsed from WB until retire stage

# CHECK-SKIP:       [0]    [1]    [2]    [3]
# CHECK-SKIP: 0.     10    3.5    0.1    0.0       addl       %eax, %eax
