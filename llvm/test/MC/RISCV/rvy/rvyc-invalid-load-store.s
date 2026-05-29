/// NOTE: Zclsd is incompatible with RVY32 since the double-width encodings are mapped to c.ly(sp)/c.sy(sp)
// RUN: not llvm-mc --triple=riscv32 --mattr=+c,+zcb,+experimental-y < %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-RVY,CHECK-RVY32 --implicit-check-not=error: %s
// RUN: not llvm-mc --triple=riscv64 --mattr=+c,+zcb,+experimental-y --defsym=RV64=1 < %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-RVY,CHECK-RVY64,CHECK-64  --implicit-check-not=error: %s
// RUN: not llvm-mc --triple=riscv32 --mattr=+c,+zcb,+zclsd,+rvy-int-mode < %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-COMPAT,CHECK-COMPAT32 --implicit-check-not=error: %s
// RUN: not llvm-mc --triple=riscv64 --mattr=+c,+zcb,+rvy-int-mode --defsym=RV64=1 < %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-COMPAT,CHECK-64 --implicit-check-not=error: %s

/// Note: Invalid operand currently has priority over missing features, so we
/// get a bad diagnostic for the *d instruction with RVY32.
/// The generated matcher code does not ignore instructions with bad requirements
/// and then ultimately ends up conflicting checks (MCK_GPRPairNoX0RV32 vs.
/// MCK_GPRNoX0) and then falls back to the generic error.

///
/// Invalid base register:
///
// CHECK: :[[#@LINE+1]]:12: error: register must be a GPR from x8 to x15
c.sb a0, 0(s5)
// CHECK: :[[#@LINE+1]]:13: error: register must be a GPR from x8 to x15
c.lbu a0, 0(s5)
// CHECK: :[[#@LINE+1]]:12: error: register must be a GPR from x8 to x15
c.lh a0, 0(s5)
// CHECK: :[[#@LINE+1]]:12: error: register must be a GPR from x8 to x15
c.sh a0, 0(s5)
// CHECK: :[[#@LINE+1]]:13: error: register must be a GPR from x8 to x15
c.lhu a0, 0(s5)
// CHECK: :[[#@LINE+1]]:12: error: register must be a GPR from x8 to x15
c.lw a0, 0(s5)
// CHECK: :[[#@LINE+1]]:12: error: register must be a GPR from x8 to x15
c.sw a0, 0(s5)
// CHECK-RVY32: :[[#@LINE+3]]:12: error: invalid operand for instruction
// CHECK-COMPAT32: :[[#@LINE+2]]:12: error: register must be a GPR from x8 to x15
// CHECK-64: :[[#@LINE+1]]:12: error: register must be a GPR from x8 to x15
c.ld a0, 0(s5)
// CHECK-RVY32: :[[#@LINE+3]]:12: error: invalid operand for instruction
// CHECK-COMPAT32: :[[#@LINE+2]]:12: error: register must be a GPR from x8 to x15
// CHECK-64: :[[#@LINE+1]]:12: error: register must be a GPR from x8 to x15
c.sd a0, 0(s5)
///
/// Invalid immediate:
///
// CHECK: :[[#@LINE+1]]:10: error: immediate must be an integer in the range [0, 3]
c.sb a0, 4(a0)
// CHECK: :[[#@LINE+1]]:11: error: immediate must be an integer in the range [0, 3]
c.lbu a0, 4(a0)
// CHECK: :[[#@LINE+1]]:10: error: immediate must be one of [0, 2]
c.sh a0, 8(a0)
// CHECK: :[[#@LINE+1]]:10: error: immediate must be one of [0, 2]
c.lh a0, 8(a0)
// CHECK: :[[#@LINE+1]]:11: error: immediate must be one of [0, 2]
c.lhu a0, 8(a0)
// CHECK: :[[#@LINE+1]]:10: error: immediate must be a multiple of 4 bytes in the range [0, 124]
c.sw a0, 7(a0)
// CHECK: :[[#@LINE+1]]:10: error: immediate must be a multiple of 4 bytes in the range [0, 124]
c.lw a0, 7(a0)
// CHECK-RVY32: :[[#@LINE+3]]:10: error: invalid operand for instruction
// CHECK-COMPAT32: :[[#@LINE+2]]:10: error: immediate must be a multiple of 8 bytes in the range [0, 248]
// CHECK-64: :[[#@LINE+1]]:10: error: immediate must be a multiple of 8 bytes in the range [0, 248]
c.sd a0, 7(a0)
// CHECK-RVY32: :[[#@LINE+3]]:10: error: invalid operand for instruction
// CHECK-COMPAT32: :[[#@LINE+2]]:10: error: immediate must be a multiple of 8 bytes in the range [0, 248]
// CHECK-64: :[[#@LINE+1]]:10: error: immediate must be a multiple of 8 bytes in the range [0, 248]
c.ld a0, 7(a0)
///
/// SP-relative loads/stores:
///
// CHECK: :[[#@LINE+1]]:15: error: register must be sp (x2)
c.lwsp a0, 16(a0)
// CHECK: :[[#@LINE+1]]:8: error: register must be a GPR excluding zero (x0)
c.lwsp x0, 16(sp)
// CHECK: :[[#@LINE+1]]:12: error: immediate must be a multiple of 4 bytes in the range [0, 252]
c.lwsp a0, 15(a0)
// CHECK: :[[#@LINE+1]]:15: error: register must be sp (x2)
c.swsp a0, 16(a0)
// CHECK: :[[#@LINE+1]]:12: error: immediate must be a multiple of 4 bytes in the range [0, 252]
c.swsp a0, 15(a0)
// CHECK-RVY32: :[[#@LINE+1]]:1: error: instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions)
c.ldsp a0, 16(sp) # valid only in compatibility mode
// CHECK-RVY32: :[[#@LINE+3]]:15: error: invalid operand for instruction
// CHECK-COMPAT32: :[[#@LINE+2]]:15: error: register must be sp (x2)
// CHECK-64: :[[#@LINE+1]]:15: error: register must be sp (x2)
c.ldsp a0, 16(a0)
// CHECK-RVY32: :[[#@LINE+3]]:8: error: invalid operand for instruction
// CHECK-COMPAT32: :[[#@LINE+2]]:8: error: invalid operand for instruction
// CHECK-64: :[[#@LINE+1]]:8: error: register must be a GPR excluding zero (x0)
c.ldsp x0, 16(sp)
// CHECK-RVY32: :[[#@LINE+3]]:12: error: invalid operand for instruction
// CHECK-COMPAT32: :[[#@LINE+2]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-64: :[[#@LINE+1]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]
c.ldsp a0, 15(a0)
// CHECK-RVY32: :[[#@LINE+3]]:15: error: invalid operand for instruction
// CHECK-COMPAT32: :[[#@LINE+2]]:15: error: register must be sp (x2)
// CHECK-64: :[[#@LINE+1]]:15: error: register must be sp (x2)
c.sdsp a0, 16(a0)
// CHECK-RVY32: :[[#@LINE+3]]:12: error: invalid operand for instruction
// CHECK-COMPAT32: :[[#@LINE+2]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-64: :[[#@LINE+1]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]
c.sdsp a0, 15(a0)

///
/// Test the new RVY instructions (illegal in compatibility mode)
/// Note: Invalid operand currently has priority over missing features, so we
/// get a bad diagnostic for compatibility mode.
///
// CHECK-COMPAT: :[[#@LINE+2]]:13: error: invalid operand for instruction
// CHECK-RVY: :[[#@LINE+1]]:13: error: register must be a GPR from x8 to x15
c.ly a0, 16(s5)
// CHECK-COMPAT: :[[#@LINE+2]]:6: error: invalid operand for instruction
// CHECK-RVY: :[[#@LINE+1]]:6: error: register must be a GPR from x8 to x15
c.ly s5, 16(a0)
// CHECK-COMPAT: :[[#@LINE+3]]:10: error: invalid operand for instruction
// CHECK-RVY32: :[[#@LINE+2]]:10: error: immediate must be a multiple of 8 bytes in the range [0, 248]
// CHECK-RVY64: :[[#@LINE+1]]:10: error: immediate must be a multiple of 16 bytes in the range [0, 496]
c.ly a0, 15(a0)
// CHECK-COMPAT: :[[#@LINE+2]]:13: error: invalid operand for instruction
// CHECK-RVY: :[[#@LINE+1]]:13: error: register must be a GPR from x8 to x15
c.sy a0, 16(s5)
// CHECK-COMPAT: :[[#@LINE+2]]:6: error: invalid operand for instruction
// CHECK-RVY: :[[#@LINE+1]]:6: error: register must be a GPR from x8 to x15
c.sy s5, 16(a0)
// CHECK-COMPAT: :[[#@LINE+3]]:10: error: invalid operand for instruction
// CHECK-RVY32: :[[#@LINE+2]]:10: error: immediate must be a multiple of 8 bytes in the range [0, 248]
// CHECK-RVY64: :[[#@LINE+1]]:10: error: immediate must be a multiple of 16 bytes in the range [0, 496]
c.sy a0, 15(a0)
// CHECK-COMPAT: :[[#@LINE+2]]:15: error: invalid operand for instruction
// CHECK-RVY: :[[#@LINE+1]]:15: error: register must be sp (x2)
c.lysp a0, 16(a0)
// CHECK-COMPAT: :[[#@LINE+2]]:8: error: invalid operand for instruction
// CHECK-RVY: :[[#@LINE+1]]:8: error: register must be a GPR excluding zero (x0)
c.lysp x0, 16(sp)
// CHECK-COMPAT: :[[#@LINE+3]]:12: error: invalid operand for instruction
// CHECK-RVY32: :[[#@LINE+2]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-RVY64: :[[#@LINE+1]]:12: error: immediate must be a multiple of 16 bytes in the range [0, 1008]
c.lysp a0, 15(sp)
// CHECK-COMPAT: :[[#@LINE+2]]:15: error: invalid operand for instruction
// CHECK-RVY: :[[#@LINE+1]]:15: error: register must be sp (x2)
c.sysp a0, 16(a0)
// CHECK-COMPAT: :[[#@LINE+3]]:12: error: invalid operand for instruction
// CHECK-RVY32: :[[#@LINE+2]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-RVY64: :[[#@LINE+1]]:12: error: immediate must be a multiple of 16 bytes in the range [0, 1008]
c.sysp a0, 15(sp)
