# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

        .macro DEF num
        int $0x\num
        .endm
        DEF 02
        DEF 08
        DEF 09
        DEF 0A
        DEF 10

# CHECK: int $2
# CHECK: int $8
# CHECK: int $9
# CHECK: int $10
# CHECK: int $16
