# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops -skip-measurements --dump-object-to-disk=0 --repetition-mode=loop --loop-body-size=1000 --result-aggregation-mode=min --opcode-name=HLT --max-configs-per-opcode=8192 | FileCheck %s

# By definition, loop repetitor can not be used for terminator instructions.
# Just check that we do not crash.

CHECK:      ---
CHECK-NEXT: mode: uops
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     HLT
