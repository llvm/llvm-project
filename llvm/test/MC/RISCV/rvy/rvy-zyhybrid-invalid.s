// RUN: llvm-mc --verify --triple riscv32 --mattr=+experimental,+experimental-zyhybrid,+c,+d < %s
// RUN: llvm-mc --verify --triple riscv64 --mattr=+experimental,+experimental-zyhybrid,+c,+d --defsym=RV64=1 < %s

ymodew x0, a0, a0 // expected-error:8 {{register must be a GPR excluding zero (x0)}}
ymodew f0, a0, a0 // expected-error:8 {{register must be a GPR excluding zero (x0)}}
yadd f0, a0, a0 // expected-error:6 {{register must be a GPR}}

.option arch, -zyhybrid
ymoder a0, a1 // expected-error:1 {{instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility)}}
ymodew a0, a1, a2 // expected-error:1 {{instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility)}}
ymodeswy // expected-error:1 {{instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility)}}
ymodeswi // expected-error:1 {{instruction requires the following: 'Zyhybrid' (Hybrid Extension for RVI Compatibility)}}
