# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=znver4 \
# RUN:   -timeline -iterations=1 < %s | FileCheck %s

# The 42 micro-ops delay dispatching the dependent vpaddd until after the
# five-cycle mask writeback is ready. It should therefore execute as soon as it
# is dispatched. If the gather's second scheduling write is dropped, vpaddd
# instead waits for the 21-cycle data result.

vpgatherdd %ymm0, (%rax,%ymm1,4), %ymm2
vpaddd %ymm0, %ymm3, %ymm4

# CHECK:      Instructions:
# CHECK:      42     21    8.25    *                   vpgatherdd
# CHECK:      Timeline view:
# CHECK:      [0,0]     {{.*}} vpgatherdd
# CHECK-NEXT: [0,1]     {{.*}}DeE{{-*}}R   vpaddd
