# REQUIRES: x86-registered-target

foo:
    nop

# RUN: llvm-mc --filetype=obj --triple=x86_64-pc-linux %s -o %t.o -g

# RUN: llvm-symbolizer 0 --relativenames --obj=%t.o \
# RUN:    | FileCheck %s -DDIR=%p --check-prefix=RELATIVENAMES

## Ensure last option wins.
# RUN: llvm-symbolizer 0 --basenames --relativenames --obj=%t.o \
# RUN:    | FileCheck %s -DDIR=%p --check-prefix=RELATIVENAMES
# RUN: llvm-symbolizer 0 --relativenames --basenames --obj=%t.o \
# RUN:    | FileCheck %s --check-prefix=BASENAMES

# RELATIVENAMES: [[DIR]]{{\\|/}}relativenames.s:4
# BASENAMES: {{^}}relativenames.s:4
