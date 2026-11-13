# RUN: llvm-mc --verify -triple riscv32 -mattr=+e,+zca < %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zca < %s \
# RUN:     | llvm-objdump --mattr=+e,+zca -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefix=CHECK-DIS %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+e,+zca < %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zca < %s \
# RUN:     | llvm-objdump --mattr=+e,+zca -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefix=CHECK-DIS %s

# Perform a simple check that registers x16-x31 (and the equivalent ABI names)
# are rejected for RV32E/RV64E, when both assembling and disassembling.

.option push
.option exact
# CHECK-DIS: 00001837 <unknown>
lui x16, 1 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 000028b7 <unknown>
lui x17, 2 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 00003937 <unknown>
lui x18, 3 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 000049b7 <unknown>
lui x19, 4 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 00005a37 <unknown>
lui x20, 5 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 00006ab7 <unknown>
lui x21, 6 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 00007b37 <unknown>
lui x22, 7 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 00008bb7 <unknown>
lui x23, 8 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 00009c37 <unknown>
lui x24, 9 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 0000acb7 <unknown>
lui x25, 10 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 0000bd37 <unknown>
lui x26, 11 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 0000cdb7 <unknown>
lui x27, 12 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 0000de37 <unknown>
lui x28, 13 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 0000eeb7 <unknown>
lui x29, 14 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 0000ff37 <unknown>
lui x30, 15 # expected-error:5 {{register must be a GPR}}
# CHECK-DIS: 00010fb7 <unknown>
lui x31, 16 # expected-error:5 {{register must be a GPR}}

# CHECK-DIS: 00011817 <unknown>
auipc a6, 17 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 00012897 <unknown>
auipc a7, 18 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 00013917 <unknown>
auipc s2, 19 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 00014997 <unknown>
auipc s3, 20 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 00015a17 <unknown>
auipc s4, 21 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 00016a97 <unknown>
auipc s5, 22 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 00017b17 <unknown>
auipc s6, 23 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 00018b97 <unknown>
auipc s7, 24 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 00019c17 <unknown>
auipc s8, 25 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 0001ac97 <unknown>
auipc s9, 26 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 0001bd17 <unknown>
auipc s10, 27 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 0001cd97 <unknown>
auipc s11, 28 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 0001de17 <unknown>
auipc t3, 29 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 0001ee97 <unknown>
auipc t4, 30 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 0001ff17 <unknown>
auipc t5, 31 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 00020f97 <unknown>
auipc t6, 32 # expected-error:7 {{register must be a GPR}}
.option pop

# CHECK-DIS: 0f81 <unknown>
c.addi x31, 0
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{register must be zero (x0)}}
# expected-note@-3:8 {{register must be a GPR excluding zero (x0)}}
# CHECK-DIS: 9846 <unknown>
c.add x16, x17 # expected-error:7 {{register must be a GPR}}
# CHECK-DIS: 8046 <unknown>
c.mv x0, x17 # expected-error:10 {{register must be a GPR excluding zero (x0)}}
