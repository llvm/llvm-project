# RUN: not llvm-mc -triple=riscv32 < %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-RV32 %s --implicit-check-not=error:
# RUN: not llvm-mc -triple=riscv64 < %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-RV64 %s --implicit-check-not=error:

lga x1, 1234
# CHECK: :[[#@LINE-1]]:9: error: operand must be a bare symbol name
lga x1, %pcrel_hi(1234)
# CHECK: :[[#@LINE-1]]:9: error: operand must be a bare symbol name
lga x1, %pcrel_lo(1234)
# CHECK: :[[#@LINE-1]]:9: error: operand must be a bare symbol name
lga x1, %pcrel_hi(foo)
# CHECK: :[[#@LINE-1]]:9: error: operand must be a bare symbol name
lga x1, %pcrel_lo(foo)
# CHECK: :[[#@LINE-1]]:9: error: operand must be a bare symbol name
lga x1, %hi(1234)
# CHECK: :[[#@LINE-1]]:9: error: operand must be a bare symbol name
lga x1, %lo(1234)
# CHECK: :[[#@LINE-1]]:9: error: operand must be a bare symbol name
lga x1, %hi(foo)
# CHECK: :[[#@LINE-1]]:9: error: operand must be a bare symbol name
lga x1, %lo(foo)
# CHECK: :[[#@LINE-1]]:9: error: operand must be a bare symbol name

sw a2, %hi(a_symbol), a3
# CHECK-RV32: :[[@LINE-1]]:8: error: operand must be a bare symbol name
# CHECK-RV64: :[[@LINE-2]]:8: error: operand must be a bare symbol name

sw a2, %lo(a_symbol), a3
# CHECK: :[[#@LINE-1]]:8: error: operand must be a bare symbol name
sw a2, %lo(a_symbol)(a4), a3
# CHECK: :[[#@LINE-1]]:27: error: expected '%' relocation specifier

# Too few operands must be rejected
sw a2, a_symbol
# CHECK: :[[#@LINE-1]]:16: error: too few operands for instruction

# Zero register as the destination/temporary for load/store pseudos is illegal
# since that would result in the auipc result being ignored.
lb x0, a_symbol
# CHECK: :[[#@LINE-1]]:4: error: register must be a GPR excluding zero (x0)
lbu x0, a_symbol
# CHECK: :[[#@LINE-1]]:5: error: register must be a GPR excluding zero (x0)
lh x0, a_symbol
# CHECK: :[[#@LINE-1]]:4: error: register must be a GPR excluding zero (x0)
lhu x0, a_symbol
# CHECK: :[[#@LINE-1]]:5: error: register must be a GPR excluding zero (x0)
lw x0, a_symbol
# CHECK: :[[#@LINE-1]]:4: error: register must be a GPR excluding zero (x0)
lwu x0, a_symbol
# CHECK-RV32: :[[#@LINE-1]]:1: error: instruction requires the following: RV64I Base Instruction Set
# CHECK-RV64: :[[#@LINE-2]]:5: error: register must be a GPR excluding zero (x0)
ld x0, a_symbol
# CHECK-RV32: :[[#@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-RV32: :[[#@LINE-2]]:1: note: instruction requires the following: RV64I Base Instruction Set
# CHECK-RV32: :[[#@LINE-3]]:1: note: instruction requires the following: 'Zilsd' (Load/Store pair instructions)
# CHECK-RV64: :[[#@LINE-4]]:4: error: register must be a GPR excluding zero (x0)

sb a2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:18: error: register must be a GPR excluding zero (x0)
sh a2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:18: error: register must be a GPR excluding zero (x0)
sw a2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:18: error: register must be a GPR excluding zero (x0)
sd a2, a_symbol, x0
# CHECK-RV32: :[[#@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK-RV32: :[[#@LINE-2]]:1: note: instruction requires the following: RV64I Base Instruction Set
# CHECK-RV32: :[[#@LINE-3]]:1: note: instruction requires the following: 'Zilsd' (Load/Store pair instructions)
# CHECK-RV64: :[[#@LINE-4]]:18: error: register must be a GPR excluding zero (x0)

.option arch, +f, +d, +q, +zfhmin
flh fa2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:20: error: register must be a GPR excluding zero (x0)
flw fa2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:20: error: register must be a GPR excluding zero (x0)
fld fa2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:20: error: register must be a GPR excluding zero (x0)
flq fa2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:20: error: register must be a GPR excluding zero (x0)
fsh fa2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:20: error: register must be a GPR excluding zero (x0)
fsw fa2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:20: error: register must be a GPR excluding zero (x0)
fsd fa2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:20: error: register must be a GPR excluding zero (x0)
fsq fa2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:20: error: register must be a GPR excluding zero (x0)
