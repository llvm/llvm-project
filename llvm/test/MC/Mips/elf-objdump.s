// 32 bit big endian
// RUN: llvm-mc -filetype=obj -triple mips-unknown-linux %s -o - | llvm-objdump -d -triple mips-unknown-linux  - | FileCheck %s
// 32 bit little endian
// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o - | llvm-objdump -d -triple mips-unknown-linux  - | FileCheck %s
// 64 bit big endian
// RUN: llvm-mc -filetype=obj -arch=mips64 -triple mips64-unknown-linux %s -o - | llvm-objdump -d -triple mips-unknown-linux - | FileCheck %s
// 64 bit little endian
// RUN: llvm-mc -filetype=obj -arch=mips64el -triple mips64el-unknown-linux %s -o - | llvm-objdump -d -triple mips-unknown-linux - | FileCheck %s

// We just want to see if llvm-objdump works at all.
// CHECK: .text
