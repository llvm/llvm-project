# RUN: llvm-mca -mtriple=aarch64 -mcpu=cortex-a55 --all-views < %s | FileCheck %s
# CHECK-NOT: Throughput Bottlenecks

# RUN: llvm-mca -mtriple=aarch64 -mcpu=cortex-a55 --bottleneck-analysis < %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-WARN
# CHECK-WARN: warning: bottleneck analysis is not supported for in-order CPU 'cortex-a55'

add      w2, w3, #1

