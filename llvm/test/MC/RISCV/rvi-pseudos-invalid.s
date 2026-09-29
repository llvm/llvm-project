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

# Zero register as the temporary for the store pseudo is also illegal
# since that would result in the auipc result being ignored.
sw a2, a_symbol, x0
# CHECK: :[[#@LINE-1]]:18: error: register must be a GPR excluding zero (x0)
