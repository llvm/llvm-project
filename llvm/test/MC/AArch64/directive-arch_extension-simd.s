// RUN: llvm-mc -triple aarch64 -mattr=-simd -filetype asm -o - %s | FileCheck %s

	.arch_extension simd

	add v0.8b, v0.8b, v0.8b
// CHECK: add v0.8b, v0.8b, v0.8b
