# RUN: not llvm-mc -triple=riscv32 -mattr=+c < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-RV32 %s --implicit-check-not="error:"
# RUN: not llvm-mc -triple=riscv64 -mattr=+c < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK %s --implicit-check-not="error:"

c.nop 0
# CHECK: :[[#@LINE-1]]:7: error: immediate must be non-zero in the range [-32, 31]

c.addi x0, 33
# CHECK: :[[#@LINE-1]]:12: error: immediate must be an integer in the range [-32, 31]

c.li x0, 42
# CHECK: :[[#@LINE-1]]:10: error: immediate must be an integer in the range [-32, 31]

c.lui x0, 0
# CHECK: :[[#@LINE-1]]:11: error: immediate must be in [0xfffe0, 0xfffff] or [1, 31]

c.mv x0, x0
# CHECK: :[[#@LINE-1]]:10: error: register must be a GPR excluding zero (x0)

c.add x0, x0
# CHECK: :[[#@LINE-1]]:11: error: register must be a GPR excluding zero (x0)

c.slli x0, 32
# CHECK-RV32: :[[#@LINE-1]]:12: error: immediate must be an integer in the range [0, 31]

c.srli64 x30
# CHECK: :[[#@LINE-1]]:10: error: register must be a GPR from x8 to x15

c.srai64 x31
# CHECK: :[[#@LINE-1]]:10: error: register must be a GPR from x8 to x15
