// RUN: not llvm-mc --triple riscv32 --mattr=+experimental-y <%s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-32 --implicit-check-not=error:
// RUN: not llvm-mc --triple riscv64 --mattr=+experimental-y <%s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-64 --implicit-check-not=error:

yaddi a0, a0, -2049
// CHECK: :[[#@LINE-1]]:15: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
yaddi a0, a0, 2048
// CHECK: :[[#@LINE-1]]:15: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
/// yadd with x0 is illegal (since that is the encoding for YMV).
/// Since X0 is invalid, we fall back to checking the immediate alias and get that error instead
/// TODO: It would be nice to report operand must be a a register other than X0 instead.
yadd a0, a1, zero
// CHECK: :[[#@LINE-1]]:14: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
yadd a0, a1, x0
// CHECK: :[[#@LINE-1]]:14: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
srliy a0, a1, 65
// CHECK-32: :[[#@LINE-1]]:15: error: immediate must be an integer equal to XLEN (32)
// CHECK-64: :[[#@LINE-2]]:15: error: immediate must be an integer equal to XLEN (64)
srliy a0, a1, 64
// CHECK-32: :[[#@LINE-1]]:15: error: immediate must be an integer equal to XLEN (32)
srliy a0, a1, 32
// CHECK-64: :[[#@LINE-1]]:15: error: immediate must be an integer equal to XLEN (64)
ybndswi a0, a0, 0
// CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]
ybndswi a0, a0, 257
// CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]
ybndswi a0, a0, 259
// CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]
ybndswi a0, a0, 508
// CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]
ybndswi a0, a0, 520
// CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]
ybndswi a0, a0, 8192
// CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]
ybndswi a0, a0, 4112
// CHECK: :[[#@LINE-1]]:17: error: immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]
