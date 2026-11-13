# RUN: llvm-mc --verify=rv32 -triple riscv32 -mattr=+xtheadmemidx < %s
# RUN: llvm-mc --verify=rv64 -triple riscv64 -mattr=+xtheadmemidx < %s

th.ldia 0(a0), (a1), 0, 0
# rv32-error@-1:1 {{instruction requires the following: RV64I Base Instruction Set}}
# rv64-error@-2:9 {{register must be a GPR}}
th.ldib		a0, 2(a1), 15, 1
# rv32-error@-1:1 {{instruction requires the following: RV64I Base Instruction Set}}
# rv64-error@-2:14 {{expected '('}}
th.lwia		a0, (a1), 30, 2
# rv32-error@-1:20 {{immediate must be an integer in the range [-16, 15]}}
# rv64-error@-2:20 {{immediate must be an integer in the range [-16, 15]}}
th.lwib		a0, (a1), -16, 43
# rv32-error@-1:25 {{immediate must be an integer in the range [0, 3]}}
# rv64-error@-2:25 {{immediate must be an integer in the range [0, 3]}}
th.lhib		a0, (a1), -17, 3
# rv32-error@-1:20 {{immediate must be an integer in the range [-16, 15]}}
# rv64-error@-2:20 {{immediate must be an integer in the range [-16, 15]}}
th.lrb		-2(a0), a1, a2, 0
# rv32-error@-1:9 {{register must be a GPR}}
# rv64-error@-2:9 {{register must be a GPR}}
th.lrw		a0, 3(a1), a2, 1
# rv32-error@-1:13 {{register must be a GPR}}
# rv64-error@-2:13 {{register must be a GPR}}
th.lrw		a0, a1, 4(a2), 2
# rv32-error@-1:17 {{register must be a GPR}}
# rv64-error@-2:17 {{register must be a GPR}}
th.lrh		a0, a1, a2, 5
# rv32-error@-1:21 {{immediate must be an integer in the range [0, 3]}}
# rv64-error@-2:21 {{immediate must be an integer in the range [0, 3]}}
th.lrhu		a0, a1, a2, -1
# rv32-error@-1:22 {{immediate must be an integer in the range [0, 3]}}
# rv64-error@-2:22 {{immediate must be an integer in the range [0, 3]}}

th.lbia  a0, (a0), 0, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lbib  a0, (a0), 0, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lbuia a0, (a0), 0, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lbuib a0, (a0), 0, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lhia  a0, (a0), 0, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lhib  a0, (a0), 0, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lhuia a0, (a0), 0, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lhuib a0, (a0), 0, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lwia  a0, (a0), 0, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lwib  a0, (a0), 0, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lwuia a0, (a0), 0, 0
# rv32-error@-1:1 {{instruction requires the following: RV64I Base Instruction Set}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lwuib a0, (a0), 0, 0
# rv32-error@-1:1 {{instruction requires the following: RV64I Base Instruction Set}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.ldia  a0, (a0), 0, 0
# rv32-error@-1:1 {{instruction requires the following: RV64I Base Instruction Set}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.ldib  a0, (a0), 0, 0
# rv32-error@-1:1 {{instruction requires the following: RV64I Base Instruction Set}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lbia  a0, (x10), 1, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
th.lbia  zero, (zero), 0, 0
# rv32-error@-1:10 {{rd and rs1 must be different}}
# rv64-error@-2:10 {{rd and rs1 must be different}}
