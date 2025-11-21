## Tests that --call-graph-info fails if .llvm.callgraph section has invalid
## value for flags field.

# REQUIRES: x86-registered-target

# RUN: llvm-mc %s -filetype=obj -triple=x86_64-pc-linux -o %t
# RUN: llvm-readelf --elf-output-style=LLVM --call-graph-info %t 2>&1 | FileCheck %s -DFILE=%t --check-prefix=WARN
# RUN: llvm-readelf --elf-output-style=JSON --pretty-print --call-graph-info %t 2>&1 | FileCheck %s -DFILE=%t --check-prefix=WARN

# WARN: warning: 'while reading call graph info's Flags': Unexpected value. Expected [0-7] but found [8]

.text
.globl _Z3foov
.type _Z3foov,@function
_Z3foov:
 callq _Z3foov@PLT

.section	.llvm.callgraph,"o",@llvm_call_graph,.text
.byte	0   #< Format version number.
.byte	8   #< Only valid values are 0 to 7
# Missing direct callees info
.text
