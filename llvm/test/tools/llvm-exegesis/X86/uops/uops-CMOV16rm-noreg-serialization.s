# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops --skip-measurements -opcode-name=CMOV16rm  -benchmarks-file=- | FileCheck %s -check-prefixes=CHECK-YAML

# https://bugs.llvm.org/show_bug.cgi?id=41448
# Verify that we correctly serialize RegNo 0 as %noreg, not as an empty string!

CHECK-YAML:      ---
CHECK-YAML-NEXT: mode:            uops
CHECK-YAML-NEXT: key:
CHECK-YAML-NEXT:   instructions:
CHECK-YAML-NEXT:     - 'CMOV16rm {{[A-Z0-9]+}} {{[A-Z0-9]+}} {{[A-Z0-9]+}} i_0x1 %noreg i_0x0 %noreg i_0x{{[0-9a-f]}}'
CHECK-YAML-LAST: ...
