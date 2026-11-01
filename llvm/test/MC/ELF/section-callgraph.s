# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-readelf -S a.o | FileCheck %s --check-prefix=PROGBITS
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: llvm-readelf -S b.o | FileCheck %s --check-prefix=PROGBITS
# RUN: llvm-mc -filetype=obj -triple=x86_64 c.s -o c.o
# RUN: llvm-readelf -S c.o | FileCheck %s --check-prefix=TYPED

## Like GNU assembler, .llvm.callgraph without a type is SHT_PROGBITS, whether or
## not "?" inherits a group. Specify %llvm_call_graph for SHT_LLVM_CALL_GRAPH.
# PROGBITS:      .llvm.callgraph   PROGBITS        0000000000000000 {{.*}} 000001 00      0   0  1
# PROGBITS:      .llvm.callgraph   PROGBITS        0000000000000000 {{.*}} 000001 00   G  0   0  1

# TYPED:         .llvm.callgraph   LLVM_CALL_GRAPH 0000000000000000 {{.*}} 000001 00      0   0  1
# TYPED:         .llvm.callgraph   LLVM_CALL_GRAPH 0000000000000000 {{.*}} 000001 00   G  0   0  1

#--- a.s
.pushsection .llvm.callgraph,""
  .byte 0
.popsection

.section .text.g,"axG",%progbits,g
g:
.pushsection .llvm.callgraph,"?"
  .byte 0
.popsection

#--- b.s
.section .text,"ax",%progbits
  ret

.pushsection .llvm.callgraph,"?"
  .byte 0
.popsection

.section .text.g,"axG",%progbits,g
g:
.pushsection .llvm.callgraph,"?"
  .byte 0
.popsection

#--- c.s
.section .text,"ax",%progbits
.pushsection .llvm.callgraph,"?",%llvm_call_graph
  .byte 0
.popsection

.section .text.g,"axG",%progbits,g
g:
.pushsection .llvm.callgraph,"?",@llvm_call_graph
  .byte 0
.popsection
