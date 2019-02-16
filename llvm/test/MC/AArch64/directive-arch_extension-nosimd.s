// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

	.arch_extension nosimd

	add v0.8b, v0.8b, v0.8b
// CHECK: error: instruction requires: neon
