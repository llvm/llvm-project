// RUN: llvm-mc --verify --triple riscv32 --mattr=+experimental-y < %s
// RUN: llvm-mc --verify --triple riscv64 --mattr=+experimental-y < %s

yaddi a0, a0, -2049 // expected-error:15 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
yaddi a0, a0, 2048 // expected-error:15 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
/// yadd with x0 is illegal (since that is the encoding for YMV).
/// Since X0 is invalid, we fall back to checking the immediate alias and get that error instead
/// TODO: It would be nice to report operand must be a a register other than X0 instead.
yadd a0, a1, zero
// expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
// expected-note@-2:14 {{register must be a GPR excluding zero (x0)}}
// expected-note@-3:14 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

yadd a0, a1, x0
// expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
// expected-note@-2:14 {{register must be a GPR excluding zero (x0)}}
// expected-note@-3:14 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

ybndswi a0, a0, 0 // expected-error:17 {{immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]}}
ybndswi a0, a0, 257 // expected-error:17 {{immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]}}
ybndswi a0, a0, 259 // expected-error:17 {{immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]}}
ybndswi a0, a0, 508 // expected-error:17 {{immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]}}
ybndswi a0, a0, 520 // expected-error:17 {{immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]}}
ybndswi a0, a0, 8192 // expected-error:17 {{immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]}}
ybndswi a0, a0, 4112 // expected-error:17 {{immediate must be an integer in the range [1, 255], a multiple of 8 in the range [256, 504], or a multiple of 16 in the range [512, 4096]}}

/// Zyhybrid instructions are not part of the base Y extension.
ymoder a0, a1 // expected-error:1 {{instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility)}}
ymodew a0, a1, a2 // expected-error:1 {{instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility)}}
ymodeswy // expected-error:1 {{instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility)}}
ymodeswi // expected-error:1 {{instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility)}}
