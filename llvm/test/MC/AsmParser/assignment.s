# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .set a, 0
TEST0:
        a = 0

# CHECK: TEST1:
# CHECK: .set b, 0
TEST1:
        .set b, 0

# CHECK: .globl	_f1
# CHECK: .set _f1, 0
        .globl _f1
        _f1 = 0

# CHECK: .globl	_f2
# CHECK: .set _f2, 0
        .globl _f2
        .set _f2, 0

