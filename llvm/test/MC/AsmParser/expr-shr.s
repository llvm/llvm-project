// RUN: llvm-mc -triple x86_64-unknown-unknown-elf %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple x86_64-unknown-darwin %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple x86_64-pc-windows-msvc %s | FileCheck %s --check-prefix=MSVC

.data

// CHECK: .quad 3

// MSVC does AShr.
// MSVC: .quad -1

.quad (~0 >> 62)
