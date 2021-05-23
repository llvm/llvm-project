// RUN: not llvm-mc -triple aarch64-unknown-none-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

// CHECK:      [[#@LINE+1]]:13: error: expected symbol name
.variant_pcs

// CHECK:      [[#@LINE+1]]:14: error: unknown symbol
.variant_pcs foo

.global foo
// CHECK:      [[#@LINE+1]]:18: error: expected newline
.variant_pcs foo bar
