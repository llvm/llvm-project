# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK: a = 0
TEST0:
        a = 0

# CHECK: TEST1:
# CHECK: b = 0
TEST1:
        b = 0

# CHECK: .globl	_f1
# CHECK: _f1 = 0
        .globl _f1
        _f1 = 0

# CHECK: .globl	_f2
# CHECK: _f2 = 0
        .globl _f2
        _f2 = 0

