# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -skip-unsupported-instructions=lack-sched %s 2>&1 | FileCheck --check-prefixes=CHECK-ALL,CHECK-SKIP %s
# RUN: not llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 %s 2>&1 | FileCheck --check-prefixes=CHECK-ALL,CHECK-ERROR %s

# Test defends that if all instructions are skipped leaving an empty input, an error is printed.

bzhi %eax, %ebx, %ecx

# CHECK-ALL-NOT: error

# CHECK-ERROR: error: found an unsupported instruction in the input assembly sequence, use -skip-unsupported-instructions=lack-sched to ignore these on the input.

# CHECK-SKIP: warning: found an unsupported instruction in the input assembly sequence, skipping with -skip-unsupported-instructions, note accuracy will be impacted:
# CHECK-SKIP: note: instruction:      bzhil   %eax, %ebx, %ecx
# CHECK-SKIP: error: no assembly instructions found.
