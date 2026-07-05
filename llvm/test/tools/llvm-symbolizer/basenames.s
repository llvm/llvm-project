# REQUIRES: x86-registered-target

foo:
    nop

# RUN: llvm-mc --filetype=obj --triple=x86_64-pc-linux %s -o %t.o -g
# RUN: llvm-symbolizer 0 --basenames --obj=%t.o | FileCheck %s
# RUN: llvm-symbolizer 0 -s --obj=%t.o | FileCheck %s
# RUN: llvm-symbolizer 0 --obj=%t.o | FileCheck %s -DDIR=%p --check-prefix=DEFAULT

# CHECK: {{^}}basenames.s:4
# DEFAULT: [[DIR]]{{\\|/}}basenames.s:4
