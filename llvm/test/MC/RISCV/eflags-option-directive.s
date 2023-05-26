# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o - | llvm-readelf -h - \
# RUN:         | FileCheck --check-prefixes=CHECK %s

# CHECK: Flags: 0x0
.option arch, +c
