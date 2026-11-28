# RUN: llvm-mc --verify -triple=riscv32 --mattr=+xsfmmbase, --mattr=+xsfmm32a8i,+xsfmm32a32f,+xsfmm32a8f,+xsfmm64a64f %s
# RUN: llvm-mc --verify -triple=riscv64 --mattr=+xsfmmbase, --mattr=+xsfmm32a8i,+xsfmm32a32f,+xsfmm32a8f,+xsfmm64a64f %s

sf.vsettnt a0, a1, e128, w1 # expected-error:20 {{operand must be e[8|16|16alt|32|64],w[1|2|4]}}

sf.vsettnt a0, a1, e8alt, w1 # expected-error:20 {{operand must be e[8|16|16alt|32|64],w[1|2|4]}}

sf.vsettnt a0, a1, e16alt, w1, a2 # expected-error:20 {{operand must be e[8|16|16alt|32|64],w[1|2|4]}}

sf.mm.f.f mt1, v8, v9 # expected-error:11 {{invalid operand for instruction}}

sf.mm.e5m2.e5m2 mt2, v8, v9 # expected-error:17 {{invalid operand for instruction}}

sf.mm.e5m2.e4m3 mt6, v8, v9 # expected-error:17 {{invalid operand for instruction}}

sf.mm.e4m3.e5m2 mt10, v8, v9 # expected-error:17 {{invalid operand for instruction}}

sf.mm.e4m3.e4m3 mt14, v8, v9 # expected-error:17 {{invalid operand for instruction}}

sf.mm.u.u mt1, v8, v9 # expected-error:11 {{invalid operand for instruction}}

sf.mm.s.u mt2, v8, v9 # expected-error:11 {{invalid operand for instruction}}

sf.mm.u.s mt3, v8, v9 # expected-error:11 {{invalid operand for instruction}}

sf.mm.s.s mt1, v8, v9 # expected-error:11 {{invalid operand for instruction}}
