# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --skip-measurements -opcode-name=SETCCr --max-configs-per-opcode=1 | FileCheck %s --check-prefix=CHECK
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --skip-measurements -opcode-name=SETCCr --max-configs-per-opcode=256 | FileCheck %s --check-prefix=SWEEP

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     'SETCCr {{.*}} i_0x{{[0-9a-f]}}'

SWEEP-DAG:      'SETCCr {{.*}} i_0x0'
SWEEP-DAG:      'SETCCr {{.*}} i_0x1'
SWEEP-DAG:      'SETCCr {{.*}} i_0x2'
SWEEP-DAG:      'SETCCr {{.*}} i_0x3'
SWEEP-DAG:      'SETCCr {{.*}} i_0x4'
SWEEP-DAG:      'SETCCr {{.*}} i_0x5'
SWEEP-DAG:      'SETCCr {{.*}} i_0x6'
SWEEP-DAG:      'SETCCr {{.*}} i_0x7'
SWEEP-DAG:      'SETCCr {{.*}} i_0x8'
SWEEP-DAG:      'SETCCr {{.*}} i_0x9'
SWEEP-DAG:      'SETCCr {{.*}} i_0xa'
SWEEP-DAG:      'SETCCr {{.*}} i_0xb'
SWEEP-DAG:      'SETCCr {{.*}} i_0xc'
SWEEP-DAG:      'SETCCr {{.*}} i_0xd'
SWEEP-DAG:      'SETCCr {{.*}} i_0xe'
SWEEP-DAG:      'SETCCr {{.*}} i_0xf'
