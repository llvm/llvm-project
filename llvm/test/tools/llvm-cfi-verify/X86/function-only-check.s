# RUN: llvm-cfi-verify %S/Inputs/function-only-check.o | FileCheck %s

# CHECK-LABEL: {{^Instruction: .* \(PROTECTED\)}}

# CHECK: Expected Protected: 1 (100.00%)
# CHECK: Unexpected Protected: 0 (0.00%)
# CHECK: Expected Unprotected: 0 (0.00%)
# CHECK: Unexpected Unprotected (BAD): 0 (0.00%)
