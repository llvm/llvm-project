# RUN: llvm-mc -filetype=obj -triple aarch64 %s -o -| llvm-readobj -h - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple aarch64-linux-gnu %s -o -| llvm-readobj -h - | FileCheck %s
# CHECK: OS/ABI: SystemV (0x0)

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-freebsd %s -o -| llvm-readobj -h - | FileCheck --check-prefix=AARCH64-FREEBSD-OSABI %s
# AARCH64-FREEBSD-OSABI: OS/ABI: FreeBSD (0x9)
