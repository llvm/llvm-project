# RUN: llvm-mc -triple x86_64-unknown-linux -filetype=obj %s -o %t
# RUN: llvm-readelf -S %t | FileCheck %s

# Tests that .llvm.callgraph section is deduced as SHT_LLVM_CALL_GRAPH
# for both ungrouped sections and grouped sections inheriting group via "?".

.section .text, "ax", %progbits
  bar:
  ret

.pushsection .llvm.callgraph, "?"
  .byte 0, 0
  .dc.a bar
  .quad 0
.popsection

.section .text.foo, "axG", %progbits, foo
  foo:
  ret

.pushsection .llvm.callgraph, "?"
  .byte 0, 0
  .dc.a foo
  .quad 0
.popsection

# CHECK: .llvm.callgraph LLVM_CALL_GRAPH {{.*}}
# CHECK: .llvm.callgraph LLVM_CALL_GRAPH {{.*}} G
