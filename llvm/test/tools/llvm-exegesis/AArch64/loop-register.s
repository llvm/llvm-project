REQUIRES: aarch64-registered-target, asserts

RUN: llvm-exegesis -mcpu=neoverse-v2 --use-dummy-perf-counters --mode=latency --debug-only=print-gen-assembly --opcode-name=ADDVv4i16v -repetition-mode=loop 2>&1 | FileCheck %s

CHECK:        0:  {{.*}}  str     x19, [sp, #-16]!
CHECK-NEXT:   4:  {{.*}}  movi    d[[REG:[0-9]+]], #0000000000000000
CHECK-NEXT:   8:  {{.*}}  mov     x19, #10000
CHECK-NEXT:   c:  {{.*}}  nop
CHECK-NEXT:   10: {{.*}}  nop
CHECK-NEXT:   14: {{.*}}  nop
CHECK-NEXT:   18: {{.*}}  nop
CHECK-NEXT:   1c: {{.*}}  nop
CHECK-NEXT:   20: {{.*}}  addv    h[[REG]], v[[REG]].4h
CHECK-NEXT:   24: {{.*}}  subs    x19, x19, #1
CHECK-NEXT:   28: {{.*}}  b.ne    #-8
CHECK-NEXT:   2c: {{.*}}  ldr     x19, [sp], #16
CHECK-NEXT:   30: {{.*}}  ret
