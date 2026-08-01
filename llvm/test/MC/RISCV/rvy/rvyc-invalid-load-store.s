/// NOTE: Zclsd is incompatible with RVY32 since the double-width encodings are mapped to c.ly(sp)/c.sy(sp)
// RUN: not llvm-mc --triple=riscv32 --mattr=+c,+zcb,+experimental-y < %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-RVY,CHECK-RVY32 --implicit-check-not=error: %s
// RUN: not llvm-mc --triple=riscv64 --mattr=+c,+zcb,+experimental-y --defsym=RV64=1 < %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-RVY,CHECK-RVY64,CHECK-64  --implicit-check-not=error: %s
// RUN: not llvm-mc --triple=riscv32 --mattr=+c,+zcb,+zclsd,+xllvmrvyipm < %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-COMPAT,CHECK-COMPAT32 --implicit-check-not=error: %s
// RUN: not llvm-mc --triple=riscv64 --mattr=+c,+zcb,+xllvmrvyipm --defsym=RV64=1 < %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,CHECK-COMPAT,CHECK-64 --implicit-check-not=error: %s

/// Note: For c.ldsp/c.sdsp, an invalid register and an invalid immediate can
/// both be independently plausible fixes, so the matcher cannot choose a
/// single "best" diagnostic and instead reports an ambiguous "invalid
/// instruction, any one of the following would fix this" error with a note
/// per candidate fix.

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
// CHECK-RVY32: :[[#@LINE+3]]:6: error: register must be a GPR from x8 to x15
// CHECK-COMPAT32: :[[#@LINE+2]]:12: error: register must be a GPR from x8 to x15
// CHECK-64: :[[#@LINE+1]]:12: error: register must be a GPR from x8 to x15
c.ld a0, 0(s5)
// CHECK-RVY32: :[[#@LINE+3]]:6: error: register must be a GPR from x8 to x15
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
// CHECK-RVY32: :[[#@LINE+3]]:6: error: register must be a GPR from x8 to x15
// CHECK-COMPAT32: :[[#@LINE+2]]:10: error: immediate must be a multiple of 8 bytes in the range [0, 248]
// CHECK-64: :[[#@LINE+1]]:10: error: immediate must be a multiple of 8 bytes in the range [0, 248]
c.sd a0, 7(a0)
// CHECK-RVY32: :[[#@LINE+3]]:6: error: register must be a GPR from x8 to x15
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
// CHECK-RVY32: :[[#@LINE+3]]:8: error: register must be a GPR excluding zero (x0)
// CHECK-COMPAT32: :[[#@LINE+2]]:15: error: register must be sp (x2)
// CHECK-64: :[[#@LINE+1]]:15: error: register must be sp (x2)
c.ldsp a0, 16(a0)
// CHECK-RVY32: :[[#@LINE+3]]:1: error: invalid instruction
// CHECK-COMPAT32: :[[#@LINE+2]]:8: error: register pair must start with an even GPR other than x0
// CHECK-64: :[[#@LINE+1]]:8: error: register must be a GPR excluding zero (x0)
c.ldsp x0, 16(sp)
// CHECK-RVY32: :[[#@LINE+9]]:1: error: invalid instruction, any one of the following would fix this:
// CHECK-RVY32-DAG: :[[#@LINE+8]]:8: note: register must be a GPR excluding zero (x0)
// CHECK-RVY32-DAG: :[[#@LINE+7]]:12: note: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-COMPAT32: :[[#@LINE+6]]:1: error: invalid instruction, any one of the following would fix this:
// CHECK-COMPAT32-DAG: :[[#@LINE+5]]:12: note: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-COMPAT32-DAG: :[[#@LINE+4]]:8: note: register must be a GPR excluding zero (x0)
// CHECK-64: :[[#@LINE+3]]:1: error: invalid instruction, any one of the following would fix this:
// CHECK-64-DAG: :[[#@LINE+2]]:12: note: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-64-DAG: :[[#@LINE+1]]:8: note: register pair must start with an even GPR other than x0
c.ldsp a0, 15(a0)
// CHECK-RVY32: :[[#@LINE+3]]:8: error: register must be a GPR
// CHECK-COMPAT32: :[[#@LINE+2]]:15: error: register must be sp (x2)
// CHECK-64: :[[#@LINE+1]]:15: error: register must be sp (x2)
c.sdsp a0, 16(a0)
// CHECK-RVY32: :[[#@LINE+9]]:1: error: invalid instruction, any one of the following would fix this:
// CHECK-RVY32-DAG: :[[#@LINE+8]]:8: note: register must be a GPR
// CHECK-RVY32-DAG: :[[#@LINE+7]]:12: note: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-COMPAT32: :[[#@LINE+6]]:1: error: invalid instruction, any one of the following would fix this:
// CHECK-COMPAT32-DAG: :[[#@LINE+5]]:12: note: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-COMPAT32-DAG: :[[#@LINE+4]]:8: note: register must be a GPR
// CHECK-64: :[[#@LINE+3]]:1: error: invalid instruction, any one of the following would fix this:
// CHECK-64-DAG: :[[#@LINE+2]]:12: note: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-64-DAG: :[[#@LINE+1]]:8: note: invalid operand for instruction
c.sdsp a0, 15(a0)

///
/// Test the new RVY instructions (illegal in compatibility mode)
/// Note: In compatibility mode the new RVY-only compressed mnemonics
/// (c.ly/c.sy/c.lysp/c.sysp) are not predicated in at all, so any use of
/// them is reported as a plain "invalid instruction" rather than an
/// operand-specific diagnostic.
///
// CHECK-COMPAT: :[[#@LINE+2]]:1: error: invalid instruction
// CHECK-RVY: :[[#@LINE+1]]:13: error: register must be a GPR from x8 to x15
c.ly a0, 16(s5)
// CHECK-COMPAT: :[[#@LINE+2]]:1: error: invalid instruction
// CHECK-RVY: :[[#@LINE+1]]:6: error: register must be a GPR from x8 to x15
c.ly s5, 16(a0)
// CHECK-COMPAT: :[[#@LINE+3]]:1: error: invalid instruction
// CHECK-RVY32: :[[#@LINE+2]]:10: error: immediate must be a multiple of 8 bytes in the range [0, 248]
// CHECK-RVY64: :[[#@LINE+1]]:10: error: immediate must be a multiple of 16 bytes in the range [0, 496]
c.ly a0, 15(a0)
// CHECK-COMPAT: :[[#@LINE+2]]:1: error: invalid instruction
// CHECK-RVY: :[[#@LINE+1]]:13: error: register must be a GPR from x8 to x15
c.sy a0, 16(s5)
// CHECK-COMPAT: :[[#@LINE+2]]:1: error: invalid instruction
// CHECK-RVY: :[[#@LINE+1]]:6: error: register must be a GPR from x8 to x15
c.sy s5, 16(a0)
// CHECK-COMPAT: :[[#@LINE+3]]:1: error: invalid instruction
// CHECK-RVY32: :[[#@LINE+2]]:10: error: immediate must be a multiple of 8 bytes in the range [0, 248]
// CHECK-RVY64: :[[#@LINE+1]]:10: error: immediate must be a multiple of 16 bytes in the range [0, 496]
c.sy a0, 15(a0)
// CHECK-COMPAT: :[[#@LINE+2]]:1: error: invalid instruction
// CHECK-RVY: :[[#@LINE+1]]:15: error: register must be sp (x2)
c.lysp a0, 16(a0)
// CHECK-COMPAT: :[[#@LINE+2]]:1: error: invalid instruction
// CHECK-RVY: :[[#@LINE+1]]:8: error: register must be a GPR excluding zero (x0)
c.lysp x0, 16(sp)
// CHECK-COMPAT: :[[#@LINE+3]]:1: error: invalid instruction
// CHECK-RVY32: :[[#@LINE+2]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-RVY64: :[[#@LINE+1]]:12: error: immediate must be a multiple of 16 bytes in the range [0, 1008]
c.lysp a0, 15(sp)
// CHECK-COMPAT: :[[#@LINE+2]]:1: error: invalid instruction
// CHECK-RVY: :[[#@LINE+1]]:15: error: register must be sp (x2)
c.sysp a0, 16(a0)
// CHECK-COMPAT: :[[#@LINE+3]]:1: error: invalid instruction
// CHECK-RVY32: :[[#@LINE+2]]:12: error: immediate must be a multiple of 8 bytes in the range [0, 504]
// CHECK-RVY64: :[[#@LINE+1]]:12: error: immediate must be a multiple of 16 bytes in the range [0, 1008]
c.sysp a0, 15(sp)
