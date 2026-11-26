## Invalid arch string

# RUN: llvm-mc --verify -triple riscv32 < %s
# RUN: llvm-mc --verify -triple riscv64 < %s

## Version strings are required for experimental extensions

.attribute arch, "rv32izicfilp" # expected-error {{invalid arch name 'rv32izicfilp', experimental extension requires explicit version number `zicfilp`}}
