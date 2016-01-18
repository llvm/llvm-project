// RUN: llvm-mc -triple x86_64-unknown-unknown-elf %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple x86_64-pc-windows-msvc %s | FileCheck %s --check-prefix=MSVC
// RUN: llvm-mc -triple x86_64-unknown-darwin %s | FileCheck %s --check-prefix=DARWIN

.data

// CHECK: .quad 3

// Both COFF and Darwin still use AShr.
// MSVC: .quad -1
// DARWIN: .quad -1

.quad (~0 >> 62)
