# RUN: llvm-mc --verify -triple riscv32 -mattr=+smrnmi < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+smrnmi < %s

mnret 0x10 # expected-error:7 {{unexpected extra operand for instruction}}
