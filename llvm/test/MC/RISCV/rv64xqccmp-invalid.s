# RUN: llvm-mc --verify -triple=riscv64 -mattr=xqccmp -M no-aliases -show-encoding < %s

qc.cm.mvsa01 a1, a2 # expected-error:14 {{register must be GPR from s0 to s7 (x8, x9 or x18-x23)}}

qc.cm.mvsa01 s0, s0 # expected-error:14 {{rs1 and rs2 must be different}}

qc.cm.mva01s a1, a2 # expected-error:14 {{register must be GPR from s0 to s7 (x8, x9 or x18-x23)}}

qc.cm.popretz {ra, s0-s10}, 112 # expected-error:15 {{invalid register list, '{ra, s0-s10}' or '{x1, x8-x9, x18-x26}' is not supported}}

qc.cm.popretz {ra, s0-s1}, 112 # expected-error:28 {{stack adjustment for register list must be a multiple of 16 bytes in the range [32, 80]}}

qc.cm.push {ra}, 16 # expected-error:18 {{stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]}}

qc.cm.pushfp {ra, s0}, 16 # expected-error:24 {{stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]}}

qc.cm.pop {ra, s0-s1}, -32 # expected-error:24 {{stack adjustment for register list must be a multiple of 16 bytes in the range [32, 80]}}

qc.cm.push {ra}, -15 # expected-error:18 {{stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]}}

qc.cm.push {ra, s0}, -15 # expected-error:22 {{stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]}}

qc.cm.pop {ra, s0-s1}, -33 # expected-error:24 {{stack adjustment for register list must be a multiple of 16 bytes in the range [32, 80]}}
