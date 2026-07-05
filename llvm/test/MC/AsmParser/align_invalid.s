# RUN: not llvm-mc -triple i386-linux-gnu < %s 2>&1 | FileCheck %s -check-prefix=ELF
# RUN: not llvm-mc -triple i386-apple-darwin < %s 2>&1 | FileCheck %s -check-prefix=DARWIN

.align 3
# ELF: error: alignment must be a power of 2
# DARWIN-NOT: error

.align 32
# ELF-NOT: error
# DARWIN: error: invalid alignment value

.align 0x100000000
# ELF:    error: alignment must be smaller than 2**32
# DARWIN: error: invalid alignment value
