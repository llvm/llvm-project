// RUN: not llvm-mc -triple aarch64-unknown-none-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

// CHECK:      [[#@LINE+1]]:13: error: expected symbol name
.variant_pcs

.global foo
// CHECK:      [[#@LINE+1]]:18: error: expected newline
.variant_pcs foo bar
