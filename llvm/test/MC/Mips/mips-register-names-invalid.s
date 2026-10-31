# RUN: not llvm-mc %s -triple=mips-unknown-freebsd -defsym O32=1 -show-encoding 2>%t0
# RUN: FileCheck %s --check-prefixes=CHECK,O32 < %t0
# RUN: not llvm-mc %s -triple=mips64-unknown-freebsd -target-abi n32 -show-encoding 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-freebsd -target-abi n64 -show-encoding 2>%t2
# RUN: FileCheck %s < %t2

# $32 used to trigger an assertion instead of the usual error message due to
# an off-by-one bug.

# CHECK: :[[@LINE+1]]:17: error: invalid register number
        add     $32, $0, $0
# CHECK: :[[@LINE+1]]:26: error: invalid register number
        lw      $8, 0x10($32)

# Names from other register classes must not be accepted as GPRs.
# CHECK: :[[@LINE+1]]:17: error: invalid operand for instruction
        add     $hi, $0, $0
# CHECK: :[[@LINE+1]]:17: error: invalid operand for instruction
        add     $hwr_cc, $0, $0
# CHECK: :[[@LINE+1]]:17: error: invalid operand for instruction
        add     $msacsr, $0, $0

# NABI-only names must not leak into O32.
.ifdef O32
# O32: :[[@LINE+1]]:17: error: invalid operand for instruction
        add     $a4, $0, $0
# O32: :[[@LINE+1]]:17: error: invalid operand for instruction
        add     $a7, $0, $0
# O32: :[[@LINE+1]]:17: error: invalid operand for instruction
        add     $kt0, $0, $0
# O32: :[[@LINE+1]]:17: error: invalid operand for instruction
        add     $kt1, $0, $0
.endif
