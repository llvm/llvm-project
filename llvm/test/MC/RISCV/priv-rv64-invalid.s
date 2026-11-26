# RUN: llvm-mc --verify -triple riscv64 < %s

hlv.wu a0, 0x10 # expected-error:17 {{expected '(' after optional integer offset}}

hlv.wu a0, a1 # expected-error:12 {{expected '(' or optional integer offset}}

hlv.wu a0, 1(a1) # expected-error:12 {{optional integer offset must be 0}}

hlv.d a0, 0x10 # expected-error:16 {{expected '(' after optional integer offset}}

hlv.d a0, a1 # expected-error:11 {{expected '(' or optional integer offset}}

hlv.d a0, 1(a1) # expected-error:11 {{optional integer offset must be 0}}

hsv.d a0, 0x10 # expected-error:16 {{expected '(' after optional integer offset}}

hsv.d a0, a1 # expected-error:11 {{expected '(' or optional integer offset}}

hsv.d a0, 1(a1) # expected-error:11 {{optional integer offset must be 0}}
