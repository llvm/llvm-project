# RUN: llvm-mc --verify -triple=riscv32 -mattr=zcmp -M no-aliases -show-encoding < %s

cm.mvsa01 a1, a2 # expected-error:11 {{register must be GPR from s0 to s7 (x8, x9 or x18-x23)}}

cm.mvsa01 s0, s0 # expected-error:11 {{rs1 and rs2 must be different}}

cm.mva01s a1, a2 # expected-error:11 {{register must be GPR from s0 to s7 (x8, x9 or x18-x23)}}

cm.popretz {ra, s0-s10}, 112 # expected-error:12 {{invalid register list, '{ra, s0-s10}' or '{x1, x8-x9, x18-x26}' is not supported}}

cm.popretz {ra, s0-s1}, 112 # expected-error:25 {{stack adjustment for register list must be a multiple of 16 bytes in the range [16, 64]}}

cm.push {ra}, 16 # expected-error:15 {{stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]}}

cm.pop {ra, s0-s1}, -32 # expected-error:21 {{stack adjustment for register list must be a multiple of 16 bytes in the range [16, 64]}}

cm.push {ra}, -8 # expected-error:15 {{stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]}}

cm.pop {s0}, -40 # expected-error:9 {{register list must start from 'ra' or 'x1'}}

cm.pop {ra, t1}, -40 # expected-error:13 {{register must be 's0'}}

cm.pop {ra, s0-t1}, -40 # expected-error:16 {{register must be in the range 's1' to 's11'}}

cm.pop {x1, x8-x9, x28}, -40 # expected-error:20 {{register must be 'x18'}}

cm.pop {x1, x8-x9, x18-x28}, -40 # expected-error:24 {{register must be in the range 'x19' to 'x27'}}

cm.pop {x1, x8-x9, x18-x17}, -40 # expected-error:24 {{register must be in the range 'x19' to 'x27'}}

cm.pop {x1, x8-f8, x18-x17}, -40 # expected-error:16 {{register must be 'x9'}}

cm.push x1, -16 # expected-error:9 {{operand must be {ra [, s0[-sN]]} or {x1 [, x8[-x9][, x18[-xN]]]}}}

cm.pop {ra, s0-f8}, -40 # expected-error:16 {{register must be in the range 's1' to 's11'}}
