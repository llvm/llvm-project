# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --skip-measurements -opcode-name=CVTSD2SI64rr -repetition-mode=loop --max-configs-per-opcode=8192 | FileCheck %s

# We used to fail to setup the snippet, and that disabled liveness tracking
# which we'd then tried to access during this run-line.
# Just check that we don't crash

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     CVTSD2SI64rr
