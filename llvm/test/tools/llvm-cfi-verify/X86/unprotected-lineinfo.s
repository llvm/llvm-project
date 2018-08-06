# RUN: llvm-mc %S/Inputs/unprotected-lineinfo.s -filetype obj \
# RUN:         -triple x86_64-linux-elf -o %t.o
# RUN: llvm-cfi-verify %t.o | FileCheck %s

# CHECK-LABEL: {{^Instruction: .* \(FAIL_BAD_CONDITIONAL_BRANCH\)}}
# CHECK-NEXT: tiny.cc:11

# CHECK: Expected Protected: 0 (0.00%)
# CHECK: Unexpected Protected: 0 (0.00%)
# CHECK: Expected Unprotected: 0 (0.00%)
# CHECK: Unexpected Unprotected (BAD): 1 (100.00%)
