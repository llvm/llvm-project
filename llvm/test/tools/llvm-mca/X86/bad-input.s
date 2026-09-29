# This test is generic but not all builders have an llvm-mca which can run natively.

# RUN: not llvm-mca -mtriple=x86_64 -mcpu=x86-64 %s -o /dev/null 2>&1 | FileCheck --check-prefixes=CHECK-ALL,CHECK %s
# RUN: not llvm-mca -mtriple=x86_64 -mcpu=x86-64 -skip-unsupported-instructions=none %s -o /dev/null 2>&1 | FileCheck --check-prefixes=CHECK-ALL,CHECK %s
# RUN: not llvm-mca -mtriple=x86_64 -mcpu=x86-64 -skip-unsupported-instructions=lack-sched %s -o /dev/null 2>&1 | FileCheck --check-prefixes=CHECK-ALL,CHECK %s
# RUN: not llvm-mca -mtriple=x86_64 -mcpu=x86-64 -skip-unsupported-instructions=parse-failure %s -o /dev/null 2>&1 | FileCheck --check-prefixes=CHECK-ALL,CHECK-SKIP %s
# RUN: not llvm-mca -mtriple=x86_64 -mcpu=x86-64 -skip-unsupported-instructions=any %s -o /dev/null 2>&1 | FileCheck --check-prefixes=CHECK-ALL,CHECK-SKIP %s

# Test checks that MCA does not produce a total cycles estimate if it encounters parse errors.

# CHECK-ALL-NOT: Total Cycles:

# CHECK: error: Assembly input parsing had errors, use -skip-unsupported-instructions=parse-failure to drop failing lines from the input.
# CHECK-SKIP: error: no assembly instructions found.

This is not a valid assembly file for any architecture (by virtue of this text.)
