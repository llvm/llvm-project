@ RUN: not llvm-mc -triple armv8 -mattr=-fp-armv8d16sp -show-encoding < %s 2>&1 | FileCheck %s

vmaxnm.f32 s4, d5, q1
@ CHECK: error: invalid instruction
vmaxnm.f64.f64 s4, d5, q1
@ CHECK: error: invalid instruction
vmaxnmge.f64.f64 s4, d5, q1
@ CHECK: error: instruction 'vmaxnm' is not predicable, but condition code specified

vcvta.s32.f32 s1, s2
@ CHECK: error: instruction requires: FPARMv8
vcvtp.u32.f32 s1, d2
@ CHECK: error: operand must be a register in range [d0, d31]
vcvtp.f32.u32 d1, q2
@ CHECK: error: invalid instruction
vcvtplo.f32.u32 s1, s2
@ CHECK: error: instruction 'vcvtp' is not predicable, but condition code specified

vrinta.f64.f64 s3, d12
@ CHECK: error: invalid instruction
vrintn.f32 d3, q12
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: operand must be a register in range [d0, d31]
@ CHECK: note: operand must be a register in range [q0, q15]
vrintz.f32 d3, q12
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: operand must be a register in range [d0, d31]
@ CHECK: note: operand must be a register in range [q0, q15]
vrintmge.f32.f32 d3, d4
@ CHECK: error: instruction 'vrintm' is not predicable, but condition code specified

aesd.8  q0, s1
@ CHECK: error: operand must be a register in range [q0, q15]
aese.8  s0, q1
@ CHECK: error: operand must be a register in range [q0, q15]
aesimc.8  s0, q1
@ CHECK: error: operand must be a register in range [q0, q15]
aesmc.8  q0, d1
@ CHECK: error: operand must be a register in range [q0, q15]
aesdge.8 q0, q1
@ CHECK: error: instruction 'aesd' is not predicable, but condition code specified

sha1h.32  d0, q1
@ CHECK: error: operand must be a register in range [q0, q15]
sha1su1.32  q0, s1
@ CHECK: error: operand must be a register in range [q0, q15]
sha256su0.32  s0, q1
@ CHECK: error: operand must be a register in range [q0, q15]
sha1heq.32  q0, q1
@ CHECK: error: instruction 'sha1h' is not predicable, but condition code specified

sha1c.32  s0, d1, q2
@ CHECK: error: invalid instruction
sha1m.32  q0, s1, q2
@ CHECK: error: operand must be a register in range [q0, q15]
sha1p.32  s0, q1, q2
@ CHECK: error: operand must be a register in range [q0, q15]
sha1su0.32  d0, q1, q2
@ CHECK: error: operand must be a register in range [q0, q15]
sha256h.32  q0, s1, q2
@ CHECK: error: operand must be a register in range [q0, q15]
sha256h2.32  q0, q1, s2
@ CHECK: error: operand must be a register in range [q0, q15]
sha256su1.32  s0, d1, q2
@ CHECK: error: invalid instruction
sha256su1lt.32  q0, d1, q2
@ CHECK: error: instruction 'sha256su1' is not predicable, but condition code specified

vmull.p64 q0, s1, s3
@ CHECK: error: invalid instruction
vmull.p64 s1, d2, d3
@ CHECK: error: operand must be a register in range [q0, q15]
vmullge.p64 q0, d16, d17
@ CHECK: error: instruction 'vmull' is not predicable, but condition code specified

// These instructions are predicable in VFP but not in NEON
vrintzeq.f32 d0, d1
vrintxgt.f32 d0, d1
@ CHECK: error: invalid operand for instruction
@ CHECK: error: invalid operand for instruction
