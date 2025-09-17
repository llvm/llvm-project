# RUN: llvm-mc -filetype=obj -mcpu=native %s 2>&1 | FileCheck %s

# CHECK-NOT: 'native' is not a recognized processor for this target (ignoring processor)
