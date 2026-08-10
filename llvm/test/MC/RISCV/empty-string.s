// REQUIRES: riscv-registered-target
// RUN: not llvm-mc -triple riscv32-unknown-linux-gnu < %s 2>&1 | FileCheck %s
"" # CHECK: error: unrecognized instruction mnemonic
