# RUN: llvm-mc %S/Inputs/protected-lineinfo.s -filetype obj \
# RUN:         -triple x86_64-linux-elf -o %t.o
# RUN: echo "src:*tiny*" > %t.blacklist.txt
# RUN: llvm-cfi-verify %t.o %t.blacklist.txt | FileCheck %s

# CHECK-LABEL: P
# CHECK-NEXT: tiny.cc:11
# CHECK-NEXT: {{^Blacklist Match:.*blacklist\.txt:1$}}
# CHECK-NEXT: ====> Unexpected Protected

# CHECK: Expected Protected: 0 (0.00%)
# CHECK: Unexpected Protected: 1 (100.00%)
# CHECK: Expected Unprotected: 0 (0.00%)
# CHECK: Unexpected Unprotected (BAD): 0 (0.00%)

# Source: (blacklist.txt):
#   src:*tiny*
