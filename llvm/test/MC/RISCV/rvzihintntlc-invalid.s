# RUN: llvm-mc --verify -triple riscv32 -mattr=+zihintntl,+c < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zihintntl,+c < %s

c.ntl.p1 1 # expected-error:10 {{unexpected extra operand for instruction}}
c.ntl.pall 2 # expected-error:12 {{unexpected extra operand for instruction}}
c.ntl.s1 3 # expected-error:10 {{unexpected extra operand for instruction}}
c.ntl.all 4 # expected-error:11 {{unexpected extra operand for instruction}}

c.ntl.p1 t0, t1 # expected-error:10 {{unexpected extra operand for instruction}}
c.ntl.pall t0, t1 # expected-error:12 {{unexpected extra operand for instruction}}
c.ntl.s1 t0, t1 # expected-error:10 {{unexpected extra operand for instruction}}
c.ntl.all t0, t1 # expected-error:11 {{unexpected extra operand for instruction}}

