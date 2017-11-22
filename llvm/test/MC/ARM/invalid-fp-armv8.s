@ RUN: not llvm-mc -triple armv8 -show-encoding -mattr=-neon < %s 2>&1 | FileCheck %s --check-prefix=V8

@ VCVT{B,T}

  vcvtt.f64.f16 d3, s1
@ V7-NOT: vcvtt.f64.f16 d3, s1      @ encoding: [0xe0,0x3b,0xb2,0xee]
  vcvtt.f16.f64 s5, d12
@ V7-NOT: vcvtt.f16.f64 s5, d12     @ encoding: [0xcc,0x2b,0xf3,0xee]

  vsel.f32 s3, s4, s6
@ V8: error: invalid instruction
  vselne.f32 s3, s4, s6
@ V8: error: invalid instruction
  vselmi.f32 s3, s4, s6
@ V8: error: invalid instruction
  vselpl.f32 s3, s4, s6
@ V8: error: invalid instruction
  vselvc.f32 s3, s4, s6
@ V8: error: invalid instruction
  vselcs.f32 s3, s4, s6
@ V8: error: invalid instruction
  vselcc.f32 s3, s4, s6
@ V8: error: invalid instruction
  vselhs.f32 s3, s4, s6
@ V8: error: invalid instruction
  vsello.f32 s3, s4, s6
@ V8: error: invalid instruction
  vselhi.f32 s3, s4, s6
@ V8: error: invalid instruction
  vsells.f32 s3, s4, s6
@ V8: error: invalid instruction
  vsellt.f32 s3, s4, s6
@ V8: error: invalid instruction
  vselle.f32 s3, s4, s6
@ V8: error: invalid instruction

vseleq.f32 s0, d2, d1
@ V8: error: invalid instruction
vselgt.f64 s3, s2, s1
@ V8: error: invalid operand for instruction
vselgt.f32 s0, q3, q1
@ V8: error: invalid instruction
vselgt.f64 q0, s3, q1
@ V8: error: invalid instruction

vmaxnm.f32 s0, d2, d1
@ V8: error: invalid instruction
vminnm.f64 s3, s2, s1
@ V8: error: invalid operand for instruction
vmaxnm.f32 s0, q3, q1
@ V8: error: invalid instruction
vmaxnm.f64 q0, s3, q1
@ V8: error: invalid instruction
vmaxnmgt.f64 q0, s3, q1
@ CHECK: error: instruction 'vmaxnm' is not predicable, but condition code specified

vcvta.s32.f64 d3, s2
@ V8: error: invalid instruction
vcvtp.s32.f32 d3, s2
@ V8: error: operand must be a register in range [s0, s31]
vcvtn.u32.f64 d3, s2
@ V8: error: invalid instruction
vcvtm.u32.f32 d3, s2
@ V8: error: operand must be a register in range [s0, s31]
vcvtnge.u32.f64 d3, s2
@ V8: error: instruction 'vcvtn' is not predicable, but condition code specified

vcvtbgt.f64.f16 q0, d3
@ V8: error: invalid instruction
vcvttlt.f64.f16 s0, s3
@ V8: error: invalid instruction, any one of the following would fix this:
@ V8: note: operand must be a register in range [d0, d31]
@ V8: note: invalid operand for instruction
vcvttvs.f16.f64 s0, s3
@ V8: error: invalid instruction, any one of the following would fix this:
@ V8: note: operand must be a register in range [d0, d31]
@ V8: note: invalid operand for instruction
vcvtthi.f16.f64 q0, d3
@ V8: error: operand must be a register in range [s0, s31]

vrintrlo.f32.f32 d3, q0
@ V8: error: invalid instruction
vrintxcs.f32.f32 d3, d0
@ V8: error: invalid instruction

vrinta.f64.f64 s3, q0
@ V8: error: invalid instruction
vrintn.f32.f32 d3, d0
@ V8: error: instruction requires: NEON
vrintp.f32 q3, q0
@ V8: error: instruction requires: NEON
vrintmlt.f32 q3, q0
@ V8: error: instruction 'vrintm' is not predicable, but condition code specified
