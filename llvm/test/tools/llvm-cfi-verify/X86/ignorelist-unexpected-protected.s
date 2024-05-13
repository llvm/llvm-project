# RUN: llvm-mc %S/Inputs/protected-lineinfo.s -filetype obj \
# RUN:         -triple x86_64-linux-elf -o %t.o
# RUN: echo "src:*tiny*" > %t.ignorelist.txt
# RUN: llvm-cfi-verify %t.o %t.ignorelist.txt | FileCheck %s

# CHECK-LABEL: {{^Instruction: .* \(PROTECTED\)}}
# CHECK-NEXT: tiny.cc:11
# CHECK-NEXT: {{^Ignorelist Match:.*ignorelist\.txt:1$}}
# CHECK-NEXT: ====> Unexpected Protected

# CHECK: Expected Protected: 0 (0.00%)
# CHECK: Unexpected Protected: 1 (100.00%)
# CHECK: Expected Unprotected: 0 (0.00%)
# CHECK: Unexpected Unprotected (BAD): 0 (0.00%)

# Source: (ignorelist.txt):
#   src:*tiny*
