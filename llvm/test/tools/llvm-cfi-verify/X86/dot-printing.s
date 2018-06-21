# RUN: llvm-mc %S/Inputs/protected-lineinfo.s -filetype obj \
# RUN:         -triple x86_64-linux-elf -o %t.o
# RUN: llvm-cfi-verify -print-graphs %t.o | FileCheck %s

# The expected output is as follows:
#   P 0x7b |  callq *%rax
#   digraph graph_0x7b {
#     "0x77:  jbe 2" -> "0x7b:  callq *%rax"
#     "0x77:  jbe 2" -> "0x79:  ud2"
#   }
#   0x7b = tiny.cc:11:3 (main)

# CHECK: {{^P.*callq +\*%rax.*$}}
# CHECK-NEXT: digraph
# CHECK-NEXT: {{^.*jbe.*->.*callq \*%rax}}
# CHECK-NEXT: {{^.*jbe.*->.*ud2}}
# CHECK-NEXT: }
# CHECK-NEXT: tiny.cc:11
