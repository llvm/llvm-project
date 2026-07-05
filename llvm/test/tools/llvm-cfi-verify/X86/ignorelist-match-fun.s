# RUN: llvm-mc %S/Inputs/unprotected-fullinfo.s -filetype obj \
# RUN:         -triple x86_64-linux-elf -o %t.o
# RUN: echo "fun:*main*" > %t.ignorelist.txt
# RUN: llvm-cfi-verify %t.o %t.ignorelist.txt | FileCheck %s

# CHECK-LABEL: {{^Instruction: .* \(FAIL_BAD_CONDITIONAL_BRANCH\)}}
# CHECK-NEXT: tiny.cc:11
# CHECK-NEXT: {{^Ignorelist Match:.*ignorelist\.txt:1$}}
# CHECK-NEXT: ====> Expected Unprotected

# CHECK: Expected Protected: 0 (0.00%)
# CHECK: Unexpected Protected: 0 (0.00%)
# CHECK: Expected Unprotected: 1 (100.00%)
# CHECK: Unexpected Unprotected (BAD): 0 (0.00%)

# Source: (ignorelist.txt):
#   fun:*main*
