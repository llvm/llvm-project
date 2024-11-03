# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zfa,+d,+zfh < %s 2>&1 | FileCheck -check-prefixes=CHECK-NO-RV32 %s
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-zfa,+d,+zfh < %s 2>&1 | FileCheck -check-prefixes=CHECK-NO-RV64 %s

# Invalid rounding modes
# CHECK-NO-RV64: error: operand must be 'rtz' floating-point rounding mode
# CHECK-NO-RV32: error: operand must be 'rtz' floating-point rounding mode
fcvtmod.w.d a1, ft1, rne

# CHECK-NO-RV64: error: operand must be 'rtz' floating-point rounding mode
# CHECK-NO-RV32: error: operand must be 'rtz' floating-point rounding mode
fcvtmod.w.d a1, ft1, dyn

# CHECK-NO-RV64: error: operand must be 'rtz' floating-point rounding mode
# CHECK-NO-RV32: error: operand must be 'rtz' floating-point rounding mode
fcvtmod.w.d a1, ft1, rmm

# CHECK-NO-RV64: error: operand must be 'rtz' floating-point rounding mode
# CHECK-NO-RV32: error: operand must be 'rtz' floating-point rounding mode
fcvtmod.w.d a1, ft1, rdn

# CHECK-NO-RV64: error: operand must be 'rtz' floating-point rounding mode
# CHECK-NO-RV32: error: operand must be 'rtz' floating-point rounding mode
fcvtmod.w.d a1, ft1, rup
