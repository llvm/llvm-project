# RUN: not llvm-mc %s -triple=mips-unknown-freebsd -show-encoding 2>%t0
# RUN: FileCheck %s < %t0

# $32 used to trigger an assertion instead of the usual error message due to
# an off-by-one bug.

# CHECK: :[[@LINE+1]]:17: error: invalid operand for instruction
        add     $32, $0, $0
