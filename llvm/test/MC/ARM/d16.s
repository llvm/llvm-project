@ RUN:     llvm-mc < %s -triple thumbv7-unknown-unknown -show-encoding -mattr=+vfp4,+d32 2>&1 | FileCheck %s --check-prefix=D32
@ RUN: not llvm-mc < %s -triple thumbv7-unknown-unknown -show-encoding -mattr=+vfp4,-d32 2>&1 | FileCheck %s --check-prefix=D16

@ D32-NOT: error:

@ D16: error: invalid instruction, any one of the following would fix this:
@ D16-NEXT: vadd.f64 d1, d2, d16
@ D16: note: operand must be a register in range [d0, d15]
@ D16: note: too many operands for instruction
vadd.f64 d1, d2, d16

@ D16: error: operand must be a register in range [d0, d15]
@ D16-NEXT: vadd.f64 d1, d17, d6
vadd.f64 d1, d17, d6

@ D16: error: operand must be a register in range [d0, d15]
@ D16-NEXT: vadd.f64 d19, d7, d6
vadd.f64 d19, d7, d6

@ D16: error: operand must be a register in range [d0, d15]
@ D16-NEXT: vcvt.f64.f32 d22, s4
vcvt.f64.f32 d22, s4

@ D16: error: operand must be a register in range [d0, d15]
@ D16-NEXT: vcvt.f32.f64 s26, d30
vcvt.f32.f64 s26, d30
