# RUN: llvm-mc --verify -triple riscv32 -mattr=+zihintntl < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zihintntl < %s
#
# ntl.* hints are always available even without Zihintntl
# (riscv-non-isa/riscv-elf-psabi-doc#474).
#
# RUN: llvm-mc --verify -triple riscv32 < %s
# RUN: llvm-mc --verify -triple riscv64 < %s

ntl.p1 1 # expected-error:8 {{unexpected extra operand for instruction}}
ntl.pall 2 # expected-error:10 {{unexpected extra operand for instruction}}
ntl.s1 3 # expected-error:8 {{unexpected extra operand for instruction}}
ntl.all 4 # expected-error:9 {{unexpected extra operand for instruction}}

ntl.p1 t0, t1 # expected-error:8 {{unexpected extra operand for instruction}}
ntl.pall t0, t1 # expected-error:10 {{unexpected extra operand for instruction}}
ntl.s1 t0, t1 # expected-error:8 {{unexpected extra operand for instruction}}
ntl.all t0, t1 # expected-error:9 {{unexpected extra operand for instruction}}
