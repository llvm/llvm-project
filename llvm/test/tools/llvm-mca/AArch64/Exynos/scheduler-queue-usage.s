# RUN: llvm-mca -march=aarch64 -mcpu=exynos-m3 -iterations=1 -verbose < %s | FileCheck %s -check-prefix=ALL
# RUN: llvm-mca -march=aarch64 -mcpu=exynos-m1 -iterations=1 -verbose < %s | FileCheck %s -check-prefix=ALL

   b   t

# ALL:      Scheduler's queue usage:
# ALL-NEXT: No scheduler resources used.
