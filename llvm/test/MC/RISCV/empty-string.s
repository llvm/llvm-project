// REQUIRES: riscv-registered-target
// RUN: llvm-mc --verify -triple riscv32-unknown-linux-gnu < %s
"" # expected-error {{unrecognized instruction mnemonic, did you mean: j, jr, la, lb, lh, li, lw, mv, or, sb, sh, sw?}}
