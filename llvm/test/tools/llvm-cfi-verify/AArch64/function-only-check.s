# RUN: llvm-cfi-verify -search-length-undef=6 %S/Inputs/function-only-check.o | FileCheck %s

# CHECK-LABEL: {{^Instruction: .* \(PROTECTED\)}}
# CHECK-NEXT: tiny.cc:9

# CHECK: Expected Protected: 1 (100.00%)
# CHECK: Unexpected Protected: 0 (0.00%)
# CHECK: Expected Unprotected: 0 (0.00%)
# CHECK: Unexpected Unprotected (BAD): 0 (0.00%)
