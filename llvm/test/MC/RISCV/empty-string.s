// REQUIRES: riscv-registered-target
// RUN: llvm-mc --verify -triple riscv32-unknown-linux-gnu < %s
"" # expected-error {{unrecognized instruction mnemonic}}
