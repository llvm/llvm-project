# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --repetition-mode=loop --x86-lbr-sample-period=521 --snippets-file=%p/Inputs/mov_add.att
# REQUIRES: exegesis-can-execute-x86_64, exegesis-can-measure-latency-lbr


CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     'MOV64ri32 RDI i_0x2'
CHECK-NEXT:     'ADD64ri8 RDI RDI i_0x10'
CHECK-NEXT: config: ''
CHECK-NEXT: {{.*}}
CHECK-NEXT: {{.*}}
CHECK-NEXT: {{.*}}
CHECK-NEXT: {{.*}}
CHECK-NEXT: num_repetitions: 10000
CHECK-NEXT: measurements:
CHECK-NEXT: {{.*}} value: 0.0001, per_snippet_value: 0.0002 {{.*}}
CHECK-LAST: ...
