# RUN: llvm-mc --verify -triple riscv32 -mattr=+xtheadsync < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+xtheadsync < %s

th.sfence.vmas t0 # expected-error:18 {{too few operands for instruction}}
th.sync t0 # expected-error:9 {{unexpected extra operand for instruction}}
th.sync.s t0 # expected-error:11 {{unexpected extra operand for instruction}}
th.sync.i t0 # expected-error:11 {{unexpected extra operand for instruction}}
th.sync.is t0 # expected-error:12 {{unexpected extra operand for instruction}}
