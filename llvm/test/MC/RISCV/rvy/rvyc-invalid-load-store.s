/// NOTE: Zclsd is incompatible with RVY32 since the double-width encodings are mapped to c.ly(sp)/c.sy(sp)
// RUN: llvm-mc --triple=riscv32 --mattr=+c,+zcb,+experimental-y --verify=expected,rvy,rvy32 %s
// RUN: llvm-mc --triple=riscv64 --mattr=+c,+zcb,+experimental-y --defsym=RV64=1 --verify=expected,rvy,rvy64,rv64 %s
// RUN: llvm-mc --triple=riscv32 --mattr=+c,+zcb,+zclsd,+experimental-zyhybrid --verify=expected,compat,compat32 %s
// RUN: llvm-mc --triple=riscv64 --mattr=+c,+zcb,+experimental-zyhybrid --defsym=RV64=1 --verify=expected,compat,rv64 %s

/// Note: For c.ldsp/c.sdsp, an invalid register and an invalid immediate can
/// both be independently plausible fixes, so the matcher cannot choose a
/// single "best" diagnostic and instead reports an ambiguous "invalid
/// instruction, any one of the following would fix this" error with a note
/// per candidate fix.

///
/// Invalid base register:
///
c.sb a0, 0(s5) // expected-error:12 {{register must be a GPR from x8 to x15}}
c.lbu a0, 0(s5) // expected-error:13 {{register must be a GPR from x8 to x15}}
c.lh a0, 0(s5) // expected-error:12 {{register must be a GPR from x8 to x15}}
c.sh a0, 0(s5) // expected-error:12 {{register must be a GPR from x8 to x15}}
c.lhu a0, 0(s5) // expected-error:13 {{register must be a GPR from x8 to x15}}
c.lw a0, 0(s5) // expected-error:12 {{register must be a GPR from x8 to x15}}
c.sw a0, 0(s5) // expected-error:12 {{register must be a GPR from x8 to x15}}
c.ld a0, 0(s5)
// rvy32-error@-1:1 {{invalid instruction}}
// compat32-error@-2:12 {{register must be a GPR from x8 to x15}}
// rv64-error@-3:12 {{register must be a GPR from x8 to x15}}
c.sd a0, 0(s5)
// rvy32-error@-1:1 {{invalid instruction}}
// compat32-error@-2:12 {{register must be a GPR from x8 to x15}}
// rv64-error@-3:12 {{register must be a GPR from x8 to x15}}

///
/// Invalid immediate:
///
c.sb a0, 4(a0) // expected-error:10 {{immediate must be an integer in the range [0, 3]}}
c.lbu a0, 4(a0) // expected-error:11 {{immediate must be an integer in the range [0, 3]}}
c.sh a0, 8(a0) // expected-error:10 {{immediate must be one of [0, 2]}}
c.lh a0, 8(a0) // expected-error:10 {{immediate must be one of [0, 2]}}
c.lhu a0, 8(a0) // expected-error:11 {{immediate must be one of [0, 2]}}
c.sw a0, 7(a0) // expected-error:10 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}
c.lw a0, 7(a0) // expected-error:10 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}
c.sd a0, 7(a0)
// rvy32-error@-1:1 {{invalid instruction}}
// compat32-error@-2:10 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}
// rv64-error@-3:10 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}
c.ld a0, 7(a0)
// rvy32-error@-1:1 {{invalid instruction}}
// compat32-error@-2:10 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}
// rv64-error@-3:10 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}

///
/// SP-relative loads/stores:
///
c.lwsp a0, 16(a0) // expected-error:15 {{register must be sp (x2)}}
c.lwsp x0, 16(sp) // expected-error:8 {{register must be a GPR excluding zero (x0)}}
c.lwsp a0, 15(a0) // expected-error:12 {{immediate must be a multiple of 4 bytes in the range [0, 252]}}
c.swsp a0, 16(a0) // expected-error:15 {{register must be sp (x2)}}
c.swsp a0, 15(a0) // expected-error:12 {{immediate must be a multiple of 4 bytes in the range [0, 252]}}
/// Valid in compatibility mode (Zclsd) and RV64, rejected on RV32Y:
c.ldsp a0, 16(sp)
// rvy32-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
// rvy32-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
// rvy32-note@-3:1 {{instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions)}}
c.ldsp a0, 16(a0)
// rvy32-error@-1:1 {{invalid instruction}}
// compat32-error@-2:15 {{register must be sp (x2)}}
// rv64-error@-3:15 {{register must be sp (x2)}}
c.ldsp x0, 16(sp)
// rvy32-error@-1:1 {{invalid instruction}}
// compat32-error@-2:8 {{register pair must start with an even GPR other than x0}}
// rv64-error@-3:8 {{register must be a GPR excluding zero (x0)}}
c.ldsp a0, 15(a0)
// rvy32-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
// rvy32-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
// rvy32-note@-3:1 {{instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions)}}
// compat32-error@-4:1 {{invalid instruction, any one of the following would fix this:}}
// compat32-note@-5:12 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}
// compat32-note@-6:1 {{instruction requires the following: RV64I Base Instruction Set}}
// rv64-error@-7:1 {{invalid instruction, any one of the following would fix this:}}
// rv64-note@-8:12 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}
// rv64-note@-9:1 {{instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions), RV32I Base Instruction Set}}
c.sdsp a0, 16(a0)
// rvy32-error@-1:1 {{invalid instruction}}
// compat32-error@-2:15 {{register must be sp (x2)}}
// rv64-error@-3:15 {{register must be sp (x2)}}
c.sdsp a0, 15(a0)
// rvy32-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
// rvy32-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
// rvy32-note@-3:1 {{instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions)}}
// compat32-error@-4:1 {{invalid instruction, any one of the following would fix this:}}
// compat32-note@-5:12 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}
// compat32-note@-6:1 {{instruction requires the following: RV64I Base Instruction Set}}
// rv64-error@-7:1 {{invalid instruction, any one of the following would fix this:}}
// rv64-note@-8:12 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}
// rv64-note@-9:1 {{instruction requires the following: 'Zclsd' (Compressed Load/Store pair instructions), RV32I Base Instruction Set}}

///
/// Test the new RVY instructions (illegal in compatibility mode)
/// Note: In compatibility mode the new RVY-only compressed mnemonics
/// (c.ly/c.sy/c.lysp/c.sysp) are not predicated in at all, so any use of
/// them is reported as a plain "invalid instruction" rather than an
/// operand-specific diagnostic.
///
c.ly a0, 16(s5)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:13 {{register must be a GPR from x8 to x15}}
c.ly s5, 16(a0)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:6 {{register must be a GPR from x8 to x15}}
c.ly a0, 15(a0)
// compat-error@-1:1 {{invalid instruction}}
// rvy32-error@-2:10 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}
// rvy64-error@-3:10 {{immediate must be a multiple of 16 bytes in the range [0, 496]}}
c.sy a0, 16(s5)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:13 {{register must be a GPR from x8 to x15}}
c.sy s5, 16(a0)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:6 {{register must be a GPR from x8 to x15}}
c.sy a0, 15(a0)
// compat-error@-1:1 {{invalid instruction}}
// rvy32-error@-2:10 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}
// rvy64-error@-3:10 {{immediate must be a multiple of 16 bytes in the range [0, 496]}}
c.lysp a0, 16(a0)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:15 {{register must be sp (x2)}}
c.lysp x0, 16(sp)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:8 {{register must be a GPR excluding zero (x0)}}
c.lysp a0, 15(sp)
// compat-error@-1:1 {{invalid instruction}}
// rvy32-error@-2:12 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}
// rvy64-error@-3:12 {{immediate must be a multiple of 16 bytes in the range [0, 1008]}}
c.sysp a0, 16(a0)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:15 {{register must be sp (x2)}}
c.sysp a0, 15(sp)
// compat-error@-1:1 {{invalid instruction}}
// rvy32-error@-2:12 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}
// rvy64-error@-3:12 {{immediate must be a multiple of 16 bytes in the range [0, 1008]}}

/// Non-GPR (e.g. FPR) registers must produce a register class diagnostic
/// rather than triggering an assertion when converting GPR to YGPR:
c.lw a0, 0(f0) // expected-error:12 {{register must be a GPR from x8 to x15}}
c.ly f0, 16(a0)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:6 {{register must be a GPR from x8 to x15}}
c.ly a0, 16(f0)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:13 {{register must be a GPR from x8 to x15}}
c.lysp f0, 16(sp)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:8 {{register must be a GPR excluding zero (x0)}}
c.lysp a0, 16(f0)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:15 {{register must be sp (x2)}}
c.sysp f0, 16(sp)
// compat-error@-1:1 {{invalid instruction}}
// rvy-error@-2:8 {{register must be a GPR}}
