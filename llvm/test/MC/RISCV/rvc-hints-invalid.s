# RUN: llvm-mc --verify=expected,rv32 -triple=riscv32 -mattr=+c < %s
# RUN: llvm-mc --verify -triple=riscv64 -mattr=+c < %s

c.nop 0
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:7 {{unexpected extra operand for instruction}}
# expected-note@-3:7 {{immediate must be non-zero in the range [-32, 31]}}

c.addi x0, 33 # expected-error:12 {{immediate must be an integer in the range [-32, 31]}}

c.li x0, 42 # expected-error:10 {{immediate must be an integer in the range [-32, 31]}}

c.lui x0, 0 # expected-error:11 {{immediate must be in [0xfffe0, 0xfffff] or [1, 31]}}

c.mv x0, x0 # expected-error:10 {{register must be a GPR excluding zero (x0)}}

c.add x0, x0 # expected-error:11 {{register must be a GPR excluding zero (x0)}}

c.slli x0, 32 # rv32-error:12 {{immediate must be an integer in the range [0, 31]}}

c.srli64 x30 # expected-error:10 {{register must be a GPR from x8 to x15}}

c.srai64 x31 # expected-error:10 {{register must be a GPR from x8 to x15}}
